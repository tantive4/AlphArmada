import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the configuration constants from your encoder to ensure the model's
# input shapes match the encoder's output shapes perfectly.
from configs import Config
from action_manager import ActionManager
from action_phase import Phase
from enum_class import *

# --- Helper Modules ---

class ResBlock(nn.Module):
    """A standard residual block for the spatial encoder's CNN."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- Main Network Architecture ---

class ArmadaNet(nn.Module):
    """
    The main neural network for the Armada AI, combining specialized encoders.
    """
    def __init__(self, action_manager: ActionManager):
        super(ArmadaNet, self).__init__()
        
        # Store the action manager to get phase-specific action space sizes
        self.action_manager = action_manager
        self.max_action_space = max(len(amap) for amap in self.action_manager.action_maps)

        # --- 1. Specialized Encoders ---

        # Scalar Encoder (MLP)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(Config.SCALAR_FEATURE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Ship Entity Encoder (Transformer)
        # The TransformerEncoderLayer handles self-attention for the entities.
        ship_encoder_layer = nn.TransformerEncoderLayer(d_model=Config.SHIP_ENTITY_FEATURE_SIZE, nhead=6, dim_feedforward=256, batch_first=True)
        self.ship_entity_encoder = nn.TransformerEncoder(ship_encoder_layer, num_layers=3)
        # A simple linear layer to get a fixed-size output after attention
        self.ship_entity_aggregator = nn.Linear(Config.SHIP_ENTITY_FEATURE_SIZE, 128)

        # Squad Entity Encoder (Transformer)
        # The TransformerEncoderLayer handles self-attention for the entities.
        squad_encoder_layer = nn.TransformerEncoderLayer(d_model=Config.SQUAD_ENTITY_FEATURE_SIZE, nhead=6, dim_feedforward=256, batch_first=True)
        self.squad_entity_encoder = nn.TransformerEncoder(squad_encoder_layer, num_layers=3)
        # A simple linear layer to get a fixed-size output after attention
        self.squad_entity_aggregator = nn.Linear(Config.SQUAD_ENTITY_FEATURE_SIZE, 128)

        
        # Spatial Encoder (ResNet)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(obstacle_type + Config.MAX_SHIPS * 2 + 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2), # downsample
            ResBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)) # Pool to a fixed 1x1 size
        )

        # Relational Encoder (CNN for the 24x24 matrix)
        self.relation_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # --- 2. Unification Torso ---
        # This combines the outputs of all encoders.
        # 64 (scalar) + 128 (ship entity) + 128 (squad entity) + 128 (spatial) + 64 (relation) = 512
        self.torso = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # --- 3. Output Heads ---

        # Value Head: Predicts the game outcome (-1 to 1)
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Tanh squashes the output to be between -1 and 1
        )
        
        # Policy Head: Predicts the probability for each possible action in a given phase.
        # We create a separate linear layer for each game phase.
        self.policy_heads = nn.ModuleDict({
            phase.name: nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, len(self.action_manager.get_action_map(phase)))
            )
            for phase in Phase if self.action_manager.get_action_map(phase)
        })

        # Auxiliary Target Heads
        self.hull_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, Config.MAX_SHIPS),
            nn.Sigmoid() # Output between 0 and 1
        )

        # Squad Head: Predicts remaining hull value for each of the 24 possible squads
        self.squad_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, Config.MAX_SQUADS),
            nn.Sigmoid() # Output between 0 and 1
        )
        
        # Game Length Head: Predicts which round the game will end on (1-6, plus 7 for games that go the distance)
        self.game_length_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Logits for a 7-class classification
        )

    def compile_fast_policy(self):
        """
        Compiles the individual ModuleDict heads into stacked tensors for 
        efficient batched inference (BMM) on MPS/CUDA.
        """
        # 1. Identify all active phases and sort them to ensure consistent indexing
        self.active_phases = sorted(
            [p for p in Phase if p.name in self.policy_heads], 
            key=lambda x: x.value
        )
        
        # 2. Create a lookup table for Phase.value -> Stack Index
        max_phase_val = max(p.value for p in self.active_phases)
        # Initialize with -1 (error value)
        self.phase_lookup = torch.full((max_phase_val + 1,), -1, dtype=torch.long, device=Config.DEVICE)
        for idx, phase in enumerate(self.active_phases):
            self.phase_lookup[phase.value] = idx

        # 3. Pre-allocate stacked weight tensors
        num_phases = len(self.active_phases)
        
        # Layer 1: 256 -> 256
        self.w1_stack = torch.zeros(num_phases, 256, 256, device=Config.DEVICE)
        self.b1_stack = torch.zeros(num_phases, 256, device=Config.DEVICE)
        
        # Layer 2: 256 -> 128
        self.w2_stack = torch.zeros(num_phases, 128, 256, device=Config.DEVICE)
        self.b2_stack = torch.zeros(num_phases, 128, device=Config.DEVICE)
        
        # Layer 3: 128 -> Action Size (Variable)
        # We pad the output dimension to max_action_space
        self.w3_stack = torch.zeros(num_phases, self.max_action_space, 128, device=Config.DEVICE)
        self.b3_stack = torch.zeros(num_phases, self.max_action_space, device=Config.DEVICE)

        # 4. Copy weights from the slow ModuleDict to the fast Stack
        with torch.no_grad():
            for idx, phase in enumerate(self.active_phases):
                head = self.policy_heads[phase.name]
                
                # Copy Layer 1 (Linear 0)
                self.w1_stack[idx] = head[0].weight
                self.b1_stack[idx] = head[0].bias
                
                # Copy Layer 2 (Linear 2)
                self.w2_stack[idx] = head[2].weight
                self.b2_stack[idx] = head[2].bias
                
                # Copy Layer 3 (Linear 4)
                # Handle variable output size by copying into the padded tensor
                out_size = head[4].out_features
                self.w3_stack[idx, :out_size, :] = head[4].weight
                self.b3_stack[idx, :out_size] = head[4].bias

        self.fast_policy_ready = True
        print(f"Fast Policy Compiled: Merged {num_phases} heads into batch tensors.")

    def forward(self, scalar_input, ship_entity_input, squad_entity_input, spatial_input, relation_input, phases):

        # --- 1. Process through Encoders (now with batch support) ---
        batch_size = scalar_input.shape[0]

        scalar_features = self.scalar_encoder(scalar_input) # [B, 64]

        # Transformer is already batch-first. No unsqueeze/squeeze needed.
        ship_entity_features_attended = self.ship_entity_encoder(ship_entity_input) # [B, MAX_SHIPS, F]
        # Aggregate over the sequence dimension (dim=1)
        ship_entity_features_mean = ship_entity_features_attended.mean(dim=1) # [B, F]
        ship_entity_features = self.ship_entity_aggregator(ship_entity_features_mean) # [B, 128]

        squad_entity_features_attended = self.squad_entity_encoder(squad_entity_input) # [B, MAX_SQUADS, F]
        # Aggregate over the sequence dimension (dim=1)
        squad_entity_features_mean = squad_entity_features_attended.mean(dim=1) # [B, F]
        squad_entity_features = self.squad_entity_aggregator(squad_entity_features_mean) # [B, 128]

        # CNNs are already batch-first. No unsqueeze needed.
        spatial_features_raw = self.spatial_encoder(spatial_input) # [B, 128, 1, 1]
        spatial_features = spatial_features_raw.view(batch_size, -1) # [B, 128]

        # Add channel dimension for CNN (dim=1)
        relation_features_raw = self.relation_encoder(relation_input.unsqueeze(1)) # [B, 64, 1, 1]
        relation_features = relation_features_raw.view(batch_size, -1) # [B, 64]


        # --- 2. Unify in Torso ---
        # Concatenate along the feature dimension (dim=1)
        combined_features = torch.cat([scalar_features, ship_entity_features, squad_entity_features, spatial_features, relation_features], dim=1)
        
        torso_output = self.torso(combined_features) # [B, 256]


        # --- 3. Get Outputs from Heads ---
        value = self.value_head(torso_output) # [B, 1]
        
        # --- Policy Head Logic ---
        
        # Check if we have compiled the fast path (Inference/MCTS mode)
        if getattr(self, 'fast_policy_ready', False):


            stack_indices = self.phase_lookup[phases]
            
            # 2. Gather weights for the specific phases in this batch
            # Shape: [B, Out_Dim, In_Dim]
            w1 = self.w1_stack[stack_indices] 
            b1 = self.b1_stack[stack_indices]
            
            w2 = self.w2_stack[stack_indices]
            b2 = self.b2_stack[stack_indices]
            
            w3 = self.w3_stack[stack_indices]
            b3 = self.b3_stack[stack_indices]
            
            # 3. Execute Unified MLP using Batch Matrix Multiplication (BMM)
            # Input x needs shape [B, 256, 1] for BMM with [B, Out, In]
            x = torso_output.unsqueeze(2) 
            
            # Layer 1
            # [B, 256, 256] @ [B, 256, 1] -> [B, 256, 1]
            x = torch.bmm(w1, x).squeeze(2) + b1
            x = F.relu(x)
            
            # Layer 2
            x = x.unsqueeze(2)
            # [B, 128, 256] @ [B, 256, 1] -> [B, 128, 1]
            x = torch.bmm(w2, x).squeeze(2) + b2
            x = F.relu(x)
            
            # Layer 3 (Final Projection)
            x = x.unsqueeze(2)
            # [B, Max_Action, 128] @ [B, 128, 1] -> [B, Max_Action, 1]
            policy_logits = torch.bmm(w3, x).squeeze(2) + b3

        else:
            # --- Fallback to Original Slow Loop (for Training/Legacy) ---
            policy_logits = torch.zeros(batch_size, self.max_action_space, device=torso_output.device)
            phase_indices = {}
            is_tensor_input = isinstance(phases, torch.Tensor)

            for i, p in enumerate(phases):
                # If input is a Tensor, we have integers (e.g. 0, 1). 
                # We must recover the Phase Name (e.g. 'COMMAND_PHASE') to key into ModuleDict.
                if is_tensor_input:
                    phase_name = Phase(p.item()).name
                else:
                    # Legacy list input
                    phase_name = p.name

                if phase_name not in phase_indices:
                    phase_indices[phase_name] = []
                phase_indices[phase_name].append(i)

            BUCKET_STEP = 32 # Round all sub-batches to nearest 32
            
            for phase_name, indices in phase_indices.items():
                # Extract the sub-batch for this phase
                group_torso_output = torso_output[indices] # Shape [N, 256]
                
                real_size = group_torso_output.shape[0]
                
                # Calculate target bucket size (e.g., 12 -> 32, 45 -> 64)
                target_size = ((real_size + BUCKET_STEP - 1) // BUCKET_STEP) * BUCKET_STEP
                
                pad_amount = target_size - real_size
                
                if pad_amount > 0:
                    # Pad the input at the end
                    # F.pad format: (left, right, top, bottom)
                    # We pad dimension 0 (bottom), so: (0, 0, 0, pad_amount)
                    padded_input = F.pad(group_torso_output, (0, 0, 0, pad_amount))
                    
                    # Pass through the layer (MPS compiles ONE graph for size Target_Size)
                    padded_logits = self.policy_heads[phase_name](padded_input)
                    
                    # Slice off the padding so gradients are correct
                    group_logits = padded_logits[:real_size]
                else:
                    # Perfect fit, no padding needed
                    group_logits = self.policy_heads[phase_name](group_torso_output)
                policy_logits[indices, :group_logits.shape[1]] = group_logits


        # --- 4. Auxiliary Heads ---
        predicted_hull = self.hull_head(torso_output)
        predicted_squads = self.squad_head(torso_output)
        predicted_game_length = self.game_length_head(torso_output)


        # --- 5. Return all outputs ---
        outputs = {
            "policy_logits": policy_logits,
            "value": value,
            "predicted_hull": predicted_hull,
            "predicted_squads": predicted_squads,
            "predicted_game_length": predicted_game_length
        }
        return outputs


# --- Example Usage (for testing the model's structure) ---
if __name__ == '__main__':
    action_manager = ActionManager()
    print(max(len(amap) for amap in action_manager.action_maps))
    # model = ArmadaNet(action_manager)
    # print(f"Model created successfully.\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # # --- Create Dummy BATCH Input Data (Batch size = 4) ---
    # B = 4
    # scalar_data = torch.randn(B, Config.SCALAR_FEATURE_SIZE)
    # ship_entity_data = torch.randn(B, Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE)
    # squad_entity_data = torch.randn(B, Config.MAX_SQUADS, Config.SQUAD_ENTITY_FEATURE_SIZE)

    # W, H = Config.BOARD_RESOLUTION

    # spatial_data = torch.randn(B, Config.MAX_SHIPS * 2, H, W)
    # relation_data = torch.randn(B, Config.MAX_SHIPS * 4, Config.MAX_SHIPS * 4)

    # # Test with a mixed batch of phases
    # current_phases = [Phase.SHIP_ACTIVATE, Phase.COMMAND_PHASE, Phase.SHIP_ACTIVATE, Phase.SHIP_DECLARE_TARGET]

    # # --- Perform a Forward Pass ---
    # model.eval()
    # with torch.no_grad():
    #     policy_output, value_output = model(scalar_data, ship_entity_data, squad_entity_data, spatial_data, relation_data, current_phases)

    # # --- Check the Output Shapes ---
    # print(f"\n--- Testing with Batch Size {B} ---")
    # print(f"Policy head output shape: {policy_output.shape} (Expected: [{B}, {model.max_action_space}])")
    # print(f"Value head output shape: {value_output.shape} (Expected: [{B}, 1])")
    # print(f"Predicted values: {value_output.squeeze().tolist()}")