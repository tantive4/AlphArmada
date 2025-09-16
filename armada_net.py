import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import the configuration constants from your encoder to ensure the model's
# input shapes match the encoder's output shapes perfectly.
from game_encoder import (
    SCALAR_FEATURE_SIZE,
    ENTITY_FEATURE_SIZE,
    MAX_SHIPS,
    BOARD_RESOLUTION,
    RELATION_FEATURE_SIZE
)
from action_space import ActionManager
from game_phase import GamePhase

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

        # --- 1. Specialized Encoders ---

        # Scalar Encoder (MLP)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(SCALAR_FEATURE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Entity Encoder (Transformer)
        # The TransformerEncoderLayer handles self-attention for the entities.
        encoder_layer = nn.TransformerEncoderLayer(d_model=ENTITY_FEATURE_SIZE, nhead=6, dim_feedforward=256, batch_first=True)
        self.entity_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # A simple linear layer to get a fixed-size output after attention
        self.entity_aggregator = nn.Linear(ENTITY_FEATURE_SIZE, 128)
        
        # Spatial Encoder (ResNet)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(MAX_SHIPS + 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
        # 64 (scalar) + 128 (entity) + 128 (spatial) + 64 (relation) = 384
        self.torso = nn.Sequential(
            nn.Linear(384, 512),
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
            phase.name: nn.Linear(256, len(self.action_manager.get_action_map(phase)['total_actions']))
            for phase in self.action_manager.action_maps.keys()
        })


    def forward(self, scalar_input, entity_input, spatial_input, relation_input, phase: GamePhase):
        # --- 1. Process through Encoders ---
        
        scalar_features = self.scalar_encoder(scalar_input)
        
        # Transformer needs a batch dimension, so we unsqueeze and squeeze.
        entity_features_attended = self.entity_encoder(entity_input.unsqueeze(0)).squeeze(0)
        # Aggregate the features of all ships into one vector using the mean.
        entity_features = self.entity_aggregator(entity_features_attended.mean(dim=0))

        # Add a batch dimension for CNNs
        spatial_features_raw = self.spatial_encoder(spatial_input.unsqueeze(0))
        spatial_features = spatial_features_raw.view(spatial_features_raw.size(0), -1).squeeze(0)

        # Add batch and channel dimensions for CNN
        relation_features_raw = self.relation_encoder(relation_input.unsqueeze(0).unsqueeze(0))
        relation_features = relation_features_raw.view(relation_features_raw.size(0), -1).squeeze(0)

        # --- 2. Unify in Torso ---
        
        # Concatenate all features into a single master vector
        combined_features = torch.cat([scalar_features, entity_features, spatial_features, relation_features], dim=0)
        
        torso_output = self.torso(combined_features)

        # --- 3. Get Outputs from Heads ---
        
        value = self.value_head(torso_output)
        
        # Select the correct policy head based on the current game phase
        policy_logits = self.policy_heads[phase.name](torso_output)
        policy_probabilities = F.softmax(policy_logits, dim=0)
        
        return policy_probabilities, value

# --- Example Usage (for testing the model's structure) ---
if __name__ == '__main__':
    # This block will only run when you execute `python model.py` directly.
    
    # Initialize the action manager to get action space sizes
    action_manager = ActionManager()

    # Create an instance of the network
    model = ArmadaNet(action_manager)
    print(f"Model created successfully.\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # --- Create Dummy Input Data (mimicking the encoder's output) ---
    # This simulates a single game state.
    scalar_data = torch.randn(SCALAR_FEATURE_SIZE)
    entity_data = torch.randn(MAX_SHIPS, ENTITY_FEATURE_SIZE)
    spatial_data = torch.randn(MAX_SHIPS + 2, BOARD_RESOLUTION, BOARD_RESOLUTION)
    relation_data = torch.randn(MAX_SHIPS * 4, MAX_SHIPS * 4) # 24x24

    # Select a phase to test, for example, the SHIP_PHASE
    current_phase = GamePhase.SHIP_PHASE

    # --- Perform a Forward Pass ---
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # We don't need to calculate gradients for this test
        policy_output, value_output = model(scalar_data, entity_data, spatial_data, relation_data, current_phase)

    # --- Check the Output Shapes ---
    expected_policy_size = len(action_manager.get_action_map(current_phase)['total_actions'])
    
    print(f"\n--- Testing with GamePhase.{current_phase.name} ---")
    print(f"Scalar input shape: {scalar_data.shape}")
    print(f"Entity input shape: {entity_data.shape}")
    print(f"Spatial input shape: {spatial_data.shape}")
    print(f"Relation input shape: {relation_data.shape}")
    print("-" * 30)
    print(f"Policy head output shape: {policy_output.shape} (Expected: {expected_policy_size})")
    print(f"Value head output shape: {value_output.shape} (Expected: torch.Size([1]))")
    print(f"Predicted value: {value_output.item()}")

    # You can uncomment this to see the raw policy logits
    print(f"Policy logits: {policy_output}")
