import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the configuration constants from your encoder to ensure the model's
# input shapes match the encoder's output shapes perfectly.
from configs import Config
from action_manager import ActionManager
from action_phase import Phase, POINTER_PHASE
from enum_class import *

# --- Helper Modules ---
class GlobalPoolingBias(nn.Module):
    """
    KataGo-style Global Pooling Bias layer.
    Computes global statistics (Mean and Max) of the feature map and 
    projects them to a channel-wise bias.
    """
    def __init__(self, channels):
        super(GlobalPoolingBias, self).__init__()
        # Input: Mean(C) + Max(C) = 2*C
        # Output: Bias(C)
        self.linear = nn.Linear(channels * 2, channels)
        
        # Initialize weights to zero so it starts as an identity operation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        
        # 1. Compute Global Stats
        # Mean Pooling
        avg_pool = x.mean(dim=(2, 3)) # [B, C]
        # Max Pooling (KataGo emphasizes this for capturing "sharp" features like single stones/units)
        max_pool = x.amax(dim=(2, 3)) # [B, C]
        
        # 2. Combine and Project
        stats = torch.cat([avg_pool, max_pool], dim=1) # [B, 2C]
        bias = self.linear(stats) # [B, C]
        
        # 3. Add Bias (Broadcast over spatial dims)
        return x + bias.unsqueeze(2).unsqueeze(3)
    
class ResBlock(nn.Module):
    """
    Pre-Activation ResBlock (KataGo Style).
    Order: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    """
    def __init__(self, channels, use_global_pooling=False):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.use_global_pooling = use_global_pooling

        # 1. Pre-Act layers
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # KataGo Global Pooling Bias (simplified implementation)
        # "Pools channels to bias other channels" -> We can stick to your SE-style approximation
        # or use exact KataGo split. Sticking to your robust GPB is fine for now.
        if self.use_global_pooling:
            self.gpb = GlobalPoolingBias(channels)

    def forward(self, x):
        # x is the input (residual path)
        
        # Branch 
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        # Global Pooling Bias applied within the branch
        if self.use_global_pooling:
            out = self.gpb(out)
            
        # Addition (Residual Connection)
        return x + out
    
class PointerHead(nn.Module):
    """
    Selects a target from a list of entities using Dot-Product Attention.
    
    Structure:
    1. Query: Generated from the Global State (Torso). 
       "I need a target that is [Weak, Close, Enemy]."
    2. Keys: Generated from each Ship's individual features.
       "I am [Weak, Close, Enemy]."
    3. Attention: Dot Product(Query, Keys) -> Softmax -> Target Index
    """
    def __init__(self, torso_dim, ship_feat_dim, internal_dim=64):
        super(PointerHead, self).__init__()

        self.query_proj = nn.Linear(torso_dim, internal_dim)
        self.key_proj = nn.Linear(ship_feat_dim, internal_dim)
        self.scale = nn.Parameter(torch.tensor(internal_dim ** -0.5))

        self.pass_head = nn.Linear(torso_dim, 1)

    def forward(self, torso_output, ship_features, valid_mask=None):
        """
        Returns:
            logits: [Batch, N + 1]
        """
        # 1. Generate Query [B, 1, Dim]
        query = self.query_proj(torso_output).unsqueeze(1)
        
        # 2. Generate Keys [B, N, Dim]
        keys = self.key_proj(ship_features)
        
        # 3. Calculate Ship Scores [B, N]
        ship_scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        ship_scores = ship_scores * self.scale
        
        # 4. Apply Masking to Ships (Optional)
        if valid_mask is not None:
            ship_scores = ship_scores.masked_fill(valid_mask == 0, -1e9)

        # 5. Calculate Pass Score [B, 1]
        pass_score = self.pass_head(torso_output)
        
        # 6. Concatenate [B, N] + [B, 1] -> [B, N + 1]
        # The 'Pass' action is now the last index.
        all_scores = torch.cat([ship_scores, pass_score], dim=1)
            
        return all_scores
    

    
# --- Main Network Architecture ---

class ArmadaNet(nn.Module):
    """
    The main neural network for the Armada AI, combining specialized encoders.
    """
    def __init__(self, action_manager: ActionManager):
        super(ArmadaNet, self).__init__()
        self.action_manager = action_manager
        self.max_action_space : int = action_manager.max_action_space

        # --- Constants & Configuration ---
        self.ship_feat_size = Config.SHIP_ENTITY_FEATURE_SIZE
        self.scalar_feat_size = Config.SCALAR_FEATURE_SIZE
        
        # --- 1. Scalar Encoder ---
        self.scalar_embed_dim = 64
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_feat_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.scalar_embed_dim),
            nn.ReLU()
        )

        # --- 1. Entity Input Projection ---
        # Projects raw ship features (size ~108) to Embedding Space (size 256)
        self.embed_dim = 256
        self.ship_embedding = nn.Linear(self.ship_feat_size, self.embed_dim)

        # --- 2. Relation Bias Network ---
        # Relation Matrix is 4x4 (16 values) per ship pair. 
        # We project this to 8 values (one bias per Attention Head).
        self.nhead = 8
        self.relation_bias_net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.nhead)
        )

        # --- 3. Transformer Encoder ---
        # d_model=256, nhead=8 (32 feat/head), dim_feedforward=512
        ship_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.nhead, 
            dim_feedforward=512, 
            batch_first=True,
            norm_first=True # Usually stabilizes deep transformers
        )
        self.ship_entity_encoder = nn.TransformerEncoder(
            ship_encoder_layer, 
            num_layers=3,
            enable_nested_tensor=False)

        # --- 4. Scatter Projectors ---
        # Output 2: Project to Presence Plane (36 channels)
        self.presence_channels = 32
        self.presence_projector = nn.Linear(self.embed_dim, self.presence_channels)
        
        # Output 3: Project to Threat Plane (12 channels)
        self.threat_channels = 4
        self.num_threat_planes = 9
        self.threat_projector = nn.Linear(self.embed_dim, self.threat_channels * self.num_threat_planes)

        # --- 5. Spatial ResNet ---
        # Input Channels: 
        #   Scattered Presence (32) + Scattered Threat (4) = 36
        self.register_buffer('bit_mask', torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8))
        self.spatial_in_channels = self.presence_channels + self.threat_channels
        self.spatial_out_channels = 64
        
        # 1. The "Head" (Initial Conv + Global Scalar Bias)
        # 5x5 Conv, Stride 2 downsampling
        self.spatial_head_conv = nn.Conv2d(self.spatial_in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False)
        
        # Projects scalar game state to bias the map
        self.spatial_global_bias = nn.Linear(self.scalar_embed_dim, 64) 
        
        # 2. The Body (6 Blocks)
        # KataGo Pattern: Regular, Regular, GP, Regular, Regular, GP
        self.spatial_trunk = nn.Sequential(
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=True),  # GP Block
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=True),  # GP Block
        )
        
        # 3. The Tail (Final Norm)
        self.spatial_tail = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # --- 6. Torso ---
        # Input = Scalar(64) + Global_Ship_State(320) + Spatial_Global(128) = 512
        self.single_ship_size = self.embed_dim + self.spatial_out_channels                              # 256 + 64 = 320
        self.spatial_global_dim = self.spatial_out_channels * 2                                         # 64 * 2 (Avg + Max) = 128
        self.torso_input_size = self.scalar_embed_dim + self.single_ship_size + self.spatial_global_dim # 64 + 320 + 128 = 512
        self.torso_output_size = 256
        
        self.torso = nn.Sequential(
            nn.Linear(self.torso_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.torso_output_size),
            nn.ReLU()
        )

        # --- 3. Output Heads ---

        # Value Head: Predicts the game outcome (-1 to 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.torso_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Tanh squashes the output to be between -1 and 1
        )
        
        # Policy Head: Predicts the probability for each possible action in a given phase.
        self.extended_torso_size = self.torso_output_size + self.single_ship_size
        # Type 1: Standard Categorical Policy Heads
        self.policy_heads = nn.ModuleDict({
            phase.name: nn.Sequential(
                nn.Linear(self.extended_torso_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, len(self.action_manager.get_action_map(phase)))
            )
            for phase in Phase if self.action_manager.get_action_map(phase) and phase not in POINTER_PHASE
        })
        # Type 2: Pointer Policy Head for selecting target ships
        self.pointer_head = PointerHead(torso_dim=self.extended_torso_size, ship_feat_dim=self.single_ship_size)


        # Auxiliary Target Heads
        self.hull_input_dim = self.single_ship_size + self.torso_output_size
        self.hull_head = nn.Sequential(
            nn.Linear(self.hull_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1), # Output 1 scalar per ship
            nn.Sigmoid() # Output between 0 and 1
        )
        
        # Game Length Head: Predicts which round the game will end on (1-6, plus 7 for games that go the distance)
        self.game_length_head = nn.Sequential(
            nn.Linear(self.torso_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Logits for a 6-class classification
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
        self.w1_stack = torch.zeros(num_phases, 256, self.extended_torso_size, device=Config.DEVICE)
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
                self.w1_stack[idx] = head[0].weight #type: ignore
                self.b1_stack[idx] = head[0].bias #type: ignore
                
                # Copy Layer 2 (Linear 2)
                self.w2_stack[idx] = head[2].weight #type: ignore
                self.b2_stack[idx] = head[2].bias #type: ignore
                
                # Copy Layer 3 (Linear 4)
                # Handle variable output size by copying into the padded tensor
                out_size = head[4].out_features # type: ignore
                self.w3_stack[idx, :out_size, :] = head[4].weight #type: ignore
                self.b3_stack[idx, :out_size] = head[4].bias #type: ignore

        self.fast_policy_ready = True

    def forward(self, scalar_input, ship_entity_input, ship_coord_input, spatial_input, relation_input, active_ship_indices, phases):
        """
        Args:
            scalar_input: [B, 45]
            ship_entity_input: [B, N, 108]
            ship_coord_input: [B, N, 2] - Normalized (0-1) coordinates for safer indexing
            spatial_input: [B, N, 10, H, W] - Plane 0 is presence, 1-9 are threat geometry
            relation_input: [B, N, N, 16] - Raw 4x4 hull relation matrix (flattened)
            phases: [B] - tensor of phases
            active_ship_indices: [B] Tensor. Int index of active ship (0 to N-1). 
                                 Use N (Config.MAX_SHIPS) or -1 for "No Active Ship".
        """
        batch_size = scalar_input.shape[0]
        N = Config.MAX_SHIPS

        # Actual Ship Mask
        valid_ship_mask = (ship_entity_input.abs().sum(dim=2) > 0)
        padding_mask = torch.zeros_like(valid_ship_mask, dtype=ship_entity_input.dtype)
        padding_mask.masked_fill_(~valid_ship_mask, float('-inf'))
        
        # === 1. Inputs ===

        # --- Scalar Encoder ---
        scalar_features = self.scalar_encoder(scalar_input)

        # --- Transformer Bias Encoder ---
        attn_bias_raw = self.relation_bias_net(relation_input)
        attn_bias = attn_bias_raw.permute(0, 3, 1, 2).reshape(batch_size * self.nhead, N, N)

        # --- Entity Transformer Encoder ---
        # here we use relation bias for each head, that applies to each ship pair.
        ship_embed = self.ship_embedding(ship_entity_input)
        ship_entity_features_attended = self.ship_entity_encoder(ship_embed, mask=attn_bias, src_key_padding_mask=padding_mask)

        # --- Spatial ResNet (Scattered Connection) ---
        # Unpack Bits on GPU: [B, N, 10, H, W_packed] -> [B, N, 10, H, W]
        unpacked_spatial = (spatial_input.unsqueeze(-1) & self.bit_mask) > 0
        unpacked_spatial = unpacked_spatial.flatten(start_dim=-2).float()

        presence_vals = self.presence_projector(ship_entity_features_attended)
        threat_vals_flat = self.threat_projector(ship_entity_features_attended)
        threat_vals = threat_vals_flat.view(batch_size, N, self.threat_channels, self.num_threat_planes)

        presence_masks = unpacked_spatial[:, :, 0]
        presence_map = torch.einsum('bnc, bnhw -> bchw', presence_vals, presence_masks)

        threat_masks = unpacked_spatial[:, :, 1:]
        threat_map = torch.einsum('bncg, bnghw -> bchw', threat_vals, threat_masks)
        spatial_combined = torch.cat([presence_map, threat_map], dim=1)
        

        # --- Spatial ResNet Execution ---
        
        # 1. Head Conv
        x = self.spatial_head_conv(spatial_combined) # [B, 64, H/2, W/2]
        
        # 2. Inject Scalar Context
        # scalar_features is [B, 64]
        # project it to bias and unsqueeze to [B, 64, 1, 1]
        global_bias = self.spatial_global_bias(scalar_features).unsqueeze(-1).unsqueeze(-1)
        x = x + global_bias
        
        # 3. Trunk & Tail
        x = self.spatial_trunk(x)
        spatial_features_map = self.spatial_tail(x)


        # --- Gather Connection (Bilinear Interpolation) ---
        # spatial_features_map: [B, C, H, W]

        # 1. Prepare Grid for grid_sample
        # [B, N, 2] -> [B, 1, N, 2], scale from [0, 1] to [-1, 1]
        grid_coords = ship_coord_input.unsqueeze(1) * 2 - 1 
        grid_coords = grid_coords.clamp(-1, 1)
        
        # 2. Sample Features
        # Input: [B, C, H, W]
        # Grid:  [B, 1, N, 2]
        # Output: [B, C, 1, N]
        if self.training and spatial_features_map.device.type == 'mps':
            gathered_spatial = F.grid_sample(
                spatial_features_map.cpu(), 
                grid_coords.cpu(), 
                mode='bilinear', 
                padding_mode='zeros', # 'zeros' is safe because we clamped coords
                align_corners=False
            ).to(spatial_features_map.device)
        else:
            gathered_spatial = F.grid_sample(
                spatial_features_map, 
                grid_coords, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
        
        # 3. Reshape to [B, N, C]
        gathered_spatial = gathered_spatial.squeeze(2).transpose(1, 2)


        # --- Ship Global State ---
        ship_combined_state = torch.cat([ship_entity_features_attended, gathered_spatial], dim=2) # [B, N, 320]
        
        # [B, N] -> [B, N, 1]
        mask_expanded = valid_ship_mask.unsqueeze(-1).float()

        # Zero out invalid ships (transformer output)
        masked_features = ship_combined_state * mask_expanded

        sum_features = masked_features.sum(dim=1) # [B, 320]
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1) # [B, 1]
        global_ship_state = sum_features / valid_counts # [B, 320]

        # --- Spatial Global Features ---
        global_pool_avg = F.adaptive_avg_pool2d(spatial_features_map, 1).flatten(1) # [B, 64]
        global_pool_max = F.adaptive_max_pool2d(spatial_features_map, 1).flatten(1) # [B, 64]

        global_spatial_state = torch.cat([global_pool_avg, global_pool_max], dim=1) # [B, 128]


        # === 2. Torso ===

        torso_input = torch.cat([scalar_features, global_ship_state, global_spatial_state], dim=1) # [B, 64 + 320 + 128 = 512]
        torso_output = self.torso(torso_input) # [B, 256]

        # -- - Extended Torso ---
        # Shape: [B, 1, 320]
        zero_ship = torch.zeros(batch_size, 1, self.single_ship_size, device=Config.DEVICE)
        ship_lookup = torch.cat([ship_combined_state, zero_ship], dim=1) # [B, N+1, 320]
        
        indices_expanded = active_ship_indices.view(batch_size, 1, 1).expand(-1, -1, self.single_ship_size)
        active_ship_features = torch.gather(ship_lookup, 1, indices_expanded).squeeze(1)

        extended_torso = torch.cat([torso_output, active_ship_features], dim=1) 


        # === 3. Output Heads ===

        # --- Value Head ---
        value = self.value_head(torso_output) # [B, 1]
        
        # --- Policy Head ---
        current_phase_val = phases[0].item()
        is_pointer_phase = (current_phase_val == Phase.SHIP_ACTIVATE) or \
                            (current_phase_val == Phase.SHIP_CHOOSE_TARGET_SHIP)
        

        if is_pointer_phase:
            # === Type 1: Pointer Policy ===
            # Selects a ship index directly using the PointerHead
            # Input: Torso "Intent" and Per-Ship "Candidates"
            policy_logits = self.pointer_head(extended_torso, ship_combined_state)


        else:
            # === Type 2: Standard Categorical Policy ===
            # Use the fast batched MLP if compiled

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
                x = extended_torso.unsqueeze(2) 
                
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
                    group_torso_output = extended_torso[indices] # Shape [N, 256]
                    
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


        # --- Auxiliary Ship Hull Head ---
        # 1. Expand Torso: [B, 256] -> [B, 1, 256] -> [B, N, 256]
        torso_expanded = torso_output.unsqueeze(1).expand(-1, N, -1)
        
        # 2. Concatenate: [B, N, 320] + [B, N, 256] -> [B, N, 576]
        hull_head_input = torch.cat([ship_combined_state, torso_expanded], dim=2)
        
        # 3. Predict: [B, N, 576] -> [B, N, 1] -> [B, N]
        predicted_hull = self.hull_head(hull_head_input).squeeze(-1)

        # --- Auxiliary Game Length Head ---
        predicted_game_length = self.game_length_head(torso_output)


        # --- 5. Return all outputs ---
        outputs = {
            "policy_logits": policy_logits,
            "value": value,
            "predicted_hull": predicted_hull,
            "predicted_game_length": predicted_game_length
        }
        return outputs
