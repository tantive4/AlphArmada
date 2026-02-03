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
    def __init__(self, policy_context_dim, ship_feat_dim, internal_dim=64):
        super(PointerHead, self).__init__()

        self.query_proj = nn.Linear(policy_context_dim, internal_dim)
        self.key_proj = nn.Linear(ship_feat_dim, internal_dim)
        self.scale = nn.Parameter(torch.tensor(internal_dim ** -0.5))

        self.pass_head = nn.Linear(policy_context_dim, 1)

    def forward(self, policy_context, ship_features, valid_mask=None):
        """
        Args:
            policy_context: [B, Dim] - The Intent (state + active ship)
            ship_features: [B, N, Dim] - The Candidates
            valid_mask: [B, N] - Boolean mask (True = Valid Ship)
        Returns:
            logits: [Batch, N + 1]
        """
        # 1. Generate Query [B, 1, Dim]
        query = self.query_proj(policy_context).unsqueeze(1)
        
        # 2. Generate Keys [B, N, Dim]
        keys = self.key_proj(ship_features)
        
        # 3. Calculate Ship Scores [B, N]
        ship_scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        ship_scores = ship_scores * self.scale
        
        # 4. Apply Masking to Ships (Optional)
        if valid_mask is not None:
            ship_scores = ship_scores.masked_fill(valid_mask == 0, -1e9)

        # 5. Calculate Pass Score [B, 1]
        pass_score = self.pass_head(policy_context)
        
        # 6. Concatenate [B, N] + [B, 1] -> [B, N + 1]
        # The 'Pass' action is now the last index.
        all_scores = torch.cat([ship_scores, pass_score], dim=1)
            
        return all_scores
    

    
# --- Main Network Architecture ---

class BigDeep(nn.Module):
    """
    Transformer-Centric "Sandwich" Architecture.
    Sequence: [Scalar_Token, Ship_1, ..., Ship_N]
    Flow: Embed -> Block1 -> Spatial Sandwich -> Block2 -> Heads
    """
    def __init__(self, action_manager: ActionManager):
        super(BigDeep, self).__init__()
        self.action_manager = action_manager
        self.max_action_space : int = action_manager.max_action_space

        # --- Constants & Configuration ---
        self.ship_feat_size = Config.SHIP_ENTITY_FEATURE_SIZE
        self.scalar_feat_size = Config.SCALAR_FEATURE_SIZE
        
        # Main Embedding Dimension (d_model)
        self.embed_dim = 256 
        self.nhead = 8

        # --- 1. Embeddings ---
        # Scalar Token Encoder (The [CLS] Token)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_feat_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim),
        )
        # Ship Entity Projection
        self.ship_embedding = nn.Linear(self.ship_feat_size, self.embed_dim)

        # --- 2. Attention Bias Parameters ---
        # The Scalar Token has no geometry. We learn its relationship to ships.
        # Shape: [Heads, 1, N] and [Heads, N, 1]
        self.relation_bias_net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.nhead)
        )
        self.bias_scale = 10.0
        
        # Shape: [Heads, 1, 1]
        self.scalar_bias_row = nn.Parameter(torch.zeros(self.nhead, 1, 1))
        self.scalar_bias_col = nn.Parameter(torch.zeros(self.nhead, 1, 1))
        self.scalar_self_bias = nn.Parameter(torch.zeros(self.nhead, 1, 1))

        # --- 3. Transformer Block 1 (Geometry Aware) ---
        # "Reasoning about immediate geometric relations"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.nhead, 
            dim_feedforward=512, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_block1 = nn.TransformerEncoder(encoder_layer, num_layers=2)


        # --- 4. Spatial Sandwich Components ---
        
        # Projectors (Transformer -> Spatial Map)
        self.presence_channels = 32
        self.presence_projector = nn.Linear(self.embed_dim, self.presence_channels)
        
        self.threat_channels = 4
        self.num_threat_planes = 9
        self.threat_projector = nn.Linear(self.embed_dim, self.threat_channels * self.num_threat_planes)

        # Spatial ResNet
        self.spatial_in_channels = self.presence_channels + self.threat_channels
        self.spatial_out_channels = 64
        self.register_buffer('bit_mask', torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8))
        
        self.spatial_head_conv = nn.Conv2d(self.spatial_in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False)
        # We still inject the raw scalar embedding into the map for global context
        self.spatial_global_bias = nn.Linear(self.embed_dim, 64) 
        
        self.spatial_trunk = nn.Sequential(
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=True),
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=False),
            ResBlock(64, use_global_pooling=True),
        )
        self.spatial_tail = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Fusion Layers (Spatial Map -> Transformer)
        # We need to merge the 64-dim spatial features back into the 256-dim token stream.
        # Scalar gets Global Pool (128 dim), Ships get Grid Sample (64 dim).
        
        # Adapt Scalar's 128 (Mean+Max) spatial stats to match ships' 64
        self.scalar_spatial_adapter = nn.Linear(self.spatial_out_channels * 2, self.spatial_out_channels) 
        
        # Main Fusion: Projects Cat(Token, Spatial) -> Token
        self.sandwich_fusion = nn.Linear(self.embed_dim + self.spatial_out_channels, self.embed_dim)

        # --- 5. Transformer Block 2 (Tactical Aware) ---
        # "Reasoning about tactical situations using spatial data"
        self.transformer_block2 = nn.TransformerEncoder(encoder_layer, num_layers=2)


        # --- 6. Output Heads ---
        
        # Policy Input = Active_Ship_Token (256) + Scalar_Token (256) = 512
        self.policy_input_dim = self.embed_dim * 2
        
        # Value Head: Uses Scalar Token (The "Game State")
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Policy Heads
        self.policy_heads = nn.ModuleDict({
            phase.name: nn.Sequential(
                nn.Linear(self.policy_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, len(self.action_manager.get_action_map(phase)))
            )
            for phase in Phase if self.action_manager.get_action_map(phase) and phase not in POINTER_PHASE
        })

        # Pointer Head: 
        # Query = Policy Context (Active + Scalar) [512]
        # Keys = Candidate Ships (Block 2 Output) [256]
        self.pointer_head = PointerHead(
            policy_context_dim=self.policy_input_dim, 
            ship_feat_dim=self.embed_dim
        )

        # Auxiliary: Hull Prediction
        # Input: Ship Token (256) + Scalar Token (256)
        self.hull_head = nn.Sequential(
            nn.Linear(self.policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary: Game Length
        # Input: Scalar Token (256)
        self.game_length_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
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
        
        # Layer 1: Input (512) -> Hidden (256)
        # Input dim is self.policy_input_dim (Scalar + Active Ship)
        self.w1_stack = torch.zeros(num_phases, 256, self.policy_input_dim, device=Config.DEVICE)
        self.b1_stack = torch.zeros(num_phases, 256, device=Config.DEVICE)
        
        # Layer 2: Hidden (256) -> Hidden (128)
        self.w2_stack = torch.zeros(num_phases, 128, 256, device=Config.DEVICE)
        self.b2_stack = torch.zeros(num_phases, 128, device=Config.DEVICE)
        
        # Layer 3: Hidden (128) -> Action Size (Variable)
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
                out_size = head[4].out_features #type: ignore
                self.w3_stack[idx, :out_size, :] = head[4].weight #type: ignore
                self.b3_stack[idx, :out_size] = head[4].bias #type: ignore

        self.fast_policy_ready = True

    def forward(self, 
                scalar_input, 
                ship_entity_input, 
                ship_coord_input, 
                spatial_input, 
                relation_input, 
                active_ship_indices, 
                phases):
        """
        Args:
            scalar_input: [B, 45]
            ship_entity_input: [B, N, 110]
            ship_coord_input: [B, N, 2] - Normalized (0-1) coordinates for safer indexing
            spatial_input: [B, N, 10, H, W] - Plane 0 is presence, 1-9 are threat geometry
            relation_input: [B, N, N, 16] - Raw 4x4 hull relation matrix (flattened)
            active_ship_indices: [B] Tensor. Int index of active ship (0 to N-1). 
                                 Use N (Config.MAX_SHIPS) or -1 for "No Active Ship".
            phases: [B] - tensor of phases
        """

        batch_size = scalar_input.shape[0]
        N = Config.MAX_SHIPS
        device = scalar_input.device

        # === 1. Embedding & Token Construction ===

        # A. Create Scalar Token (The "Game State" / [CLS] Token)
        # [B, 45] -> [B, 1, 256]
        scalar_token = self.scalar_encoder(scalar_input).unsqueeze(1)

        # B. Create Ship Tokens
        # [B, N, 108] -> [B, N, 256]
        ship_tokens = self.ship_embedding(ship_entity_input)

        # C. Combine into Single Sequence
        # Shape: [B, N+1, 256] (Index 0 is Scalar, 1..N are Ships)
        tokens = torch.cat([scalar_token, ship_tokens], dim=1)

        # === 2. Attention Mask & Bias Construction ===

        # A. Padding Mask (Who is real?)
        # [B, N] - True if ship exists
        valid_ship_mask = (ship_entity_input.abs().sum(dim=2) > 0)
        
        # [B, N+1] - Initialize with False (Keep all)
        # In PyTorch MultiheadAttention, True values are IGNORED.
        src_key_padding_mask = torch.zeros(batch_size, N + 1, dtype=torch.bool, device=device)
        
        # Mask out invalid ships (indices 1 to N). Index 0 (Scalar) is always kept.
        src_key_padding_mask[:, 1:] = ~valid_ship_mask

        # B. Attention Bias (Who relates to whom?)
        # 1. Process Geometric Relations (Ship-to-Ship)
        # [B, N, N, 16] -> [B, N, N, Heads] -> [B, Heads, N, N]
        geom_bias = self.relation_bias_net(relation_input)
        geom_bias = torch.tanh(geom_bias) * self.bias_scale
        geom_bias = geom_bias.permute(0, 3, 1, 2) 

        # 2. Construct Full Bias Matrix [B, Heads, N+1, N+1]
        # Init Full Bias
        full_bias = torch.zeros(batch_size, self.nhead, N + 1, N + 1, device=device)
        full_bias[:, :, 1:, 1:] = geom_bias

        # Bias 2: Structural Scalar Bias (Broadcasted)
        # Scalar -> Ships (Row 0, Cols 1..N)
        # Use single param [Heads, 1, 1] expanded to [B, Heads, 1, N]
        full_bias[:, :, 0:1, 1:] = self.scalar_bias_row.unsqueeze(0).expand(batch_size, -1, -1, N)
        
        # Ships -> Scalar (Rows 1..N, Col 0)
        # Use single param [Heads, 1, 1] expanded to [B, Heads, N, 1]
        full_bias[:, :, 1:, 0:1] = self.scalar_bias_col.unsqueeze(0).expand(batch_size, -1, N, -1)
        
        # Scalar -> Scalar (0, 0)
        full_bias[:, :, 0, 0] = self.scalar_self_bias.squeeze().unsqueeze(0).expand(batch_size, -1)

        # 4. Reshape for Transformer [B*Heads, N+1, N+1]
        attn_bias = full_bias.reshape(batch_size * self.nhead, N + 1, N + 1)


        # === 3. Transformer Block 1 (Geometric Reasoning) ===
        # Input: [B, N+1, 256]
        # Output: [B, N+1, 256]
        tokens_l1 = self.transformer_block1(tokens, mask=attn_bias, src_key_padding_mask=src_key_padding_mask)

        # === 4. Spatial Sandwich (Scatter -> ResNet -> Gather) ===

        # A. Scatter (Transformer -> Map)
        # Extract only ship tokens for projection
        ships_l1 = tokens_l1[:, 1:] # [B, N, 256]

        # Unpack Spatial Input bits [B, N, 10, H, W]
        unpacked_spatial = (spatial_input.unsqueeze(-1) & self.bit_mask) > 0
        unpacked_spatial = unpacked_spatial.flatten(start_dim=-2).float() # [B, N, 10, H, W]

        # Project Ships to Presence/Threat values
        presence_vals = self.presence_projector(ships_l1) # [B, N, 32]
        threat_vals = self.threat_projector(ships_l1).view(batch_size, N, self.threat_channels, self.num_threat_planes) # [B, N, 4, 9]

        # Combine Values with Masks (Scatter Sum)
        # [B, N, C] * [B, N, H, W] -> Sum(N) -> [B, C, H, W]
        presence_map = torch.einsum('bnc, bnhw -> bchw', presence_vals, unpacked_spatial[:, :, 0])
        threat_map = torch.einsum('bncg, bnghw -> bchw', threat_vals, unpacked_spatial[:, :, 1:])
        
        spatial_combined = torch.cat([presence_map, threat_map], dim=1) # [B, 36, H, W]

        # B. ResNet Processing
        # 1. Head
        x = self.spatial_head_conv(spatial_combined) # [B, 64, H/2, W/2]
        
        # 2. Inject Raw Global Context (Standard embedding, not the token) into spatial map
        # This helps the map know "It is round 6" globally.
        # Use the raw embedding from token generation (squeezed) or re-project scalar input.
        # Using tokens_l1[:, 0] is also possible, but raw embedding is cleaner for "pure context".
        scalar_embed_raw = scalar_token.squeeze(1) # [B, 256]
        global_bias = self.spatial_global_bias(scalar_embed_raw).unsqueeze(-1).unsqueeze(-1)
        x = x + global_bias

        # 3. Trunk
        x = self.spatial_trunk(x)
        spatial_features_map = self.spatial_tail(x) # [B, 64, H/2, W/2]


        # C. Gather (Map -> Transformer)
        
        # 1. Gather for Ships (Grid Sample)
        # [B, N, 2] -> [B, 1, N, 2] -> Sample -> [B, 64, 1, N] -> [B, N, 64]
        grid_coords = ship_coord_input.unsqueeze(1) * 2 - 1 
        grid_coords = grid_coords.clamp(-1, 1)
        
        if self.training and device == 'mps':
             # MPS workaround for grid_sample
             gathered_spatial = F.grid_sample(
                spatial_features_map.cpu(), grid_coords.cpu(), align_corners=False, padding_mode='zeros'
            ).to(device)
        else:
            gathered_spatial = F.grid_sample(
                spatial_features_map, grid_coords, align_corners=False, padding_mode='zeros'
            )
        ship_spatial_ctx = gathered_spatial.squeeze(2).transpose(1, 2) # [B, N, 64]

        # 2. Gather for Scalar Token (Global Pool)
        # The Game State needs to know "Is the map busy? Is it dangerous?"
        # Avg + Max Pool [B, 64, H, W] -> [B, 128]
        sp_avg = F.adaptive_avg_pool2d(spatial_features_map, 1).flatten(1)
        sp_max = F.adaptive_max_pool2d(spatial_features_map, 1).flatten(1)
        scalar_spatial_raw = torch.cat([sp_avg, sp_max], dim=1) 
        
        # Project 128 -> 64 to match ship spatial dimension
        scalar_spatial_ctx = self.scalar_spatial_adapter(scalar_spatial_raw).unsqueeze(1) # [B, 1, 64]

        # 3. Concatenate Spatial Context
        # Scalar: [B, 1, 64], Ships: [B, N, 64] -> [B, N+1, 64]
        spatial_ctx = torch.cat([scalar_spatial_ctx, ship_spatial_ctx], dim=1)

        # D. Sandwich Fusion
        # Concatenate [Original_Token (256), Spatial_Context (64)] -> 320
        # Project back to 256
        fused_input = torch.cat([tokens_l1, spatial_ctx], dim=2)
        tokens_l2_input = self.sandwich_fusion(fused_input) # [B, N+1, 256]


        # === 5. Transformer Block 2 (Tactical Reasoning) ===
        # Reuse the same attention bias (or you could have a separate one)
        final_tokens = self.transformer_block2(tokens_l2_input, mask=attn_bias, src_key_padding_mask=src_key_padding_mask)


        # === 6. Head Preparation ===
        
        # Separate the "Brains"
        scalar_final_state = final_tokens[:, 0] # [B, 256]
        all_ship_final_states = final_tokens[:, 1:] # [B, N, 256]

        # Zero out invalid ships in the output to be safe
        all_ship_final_states = all_ship_final_states * valid_ship_mask.unsqueeze(-1).float()

        # Identify Active Ship for Policy
        # active_ship_indices is [B]. We need to gather [B, N, 256] -> [B, 256]
        # Handle "No Active Ship" case (index -1 or N) if necessary by padding
        # Assuming active_ship_indices are valid 0..N-1 for phases that need it.
        
        # Create a dummy zero ship at index N for gathering invalid indices
        zero_ship = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        lookup_ships = torch.cat([all_ship_final_states, zero_ship], dim=1)
        
        # Gather active ship features
        # Clamp indices to handle -1 or other indicators
        gather_indices = active_ship_indices.view(batch_size, 1, 1).expand(-1, -1, self.embed_dim)
        active_ship_state = torch.gather(lookup_ships, 1, gather_indices).squeeze(1)

        # Construct Policy Context: [Active_Ship (256) + Game_State (256)]
        policy_context = torch.cat([active_ship_state, scalar_final_state], dim=1) # [B, 512]








        # === 3. Output Heads ===

        # --- Value Head ---
        value = self.value_head(scalar_final_state) # [B, 1]
        
        # --- Policy Head ---
        current_phase_val = phases[0].item()
        is_pointer_phase = (current_phase_val == Phase.SHIP_ACTIVATE) or \
                            (current_phase_val == Phase.SHIP_CHOOSE_TARGET_SHIP)
        

        if is_pointer_phase:
            # === Type 1: Pointer Policy ===
            # Selects a ship index directly using the PointerHead
            # Input: Torso "Intent" and Per-Ship "Candidates"
            policy_logits = self.pointer_head(
                policy_context, 
                all_ship_final_states, 
                valid_mask=valid_ship_mask
            )


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
                x = policy_context.unsqueeze(2) 
                
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
                policy_logits = torch.zeros(batch_size, self.max_action_space, device=device)
                phase_indices = {}

                for i, p in enumerate(phases):
                    # If input is a Tensor, we have integers (e.g. 0, 1). 
                    # We must recover the Phase Name (e.g. 'COMMAND_PHASE') to key into ModuleDict.
                    phase_name = Phase(p.item()).name

                    if phase_name not in phase_indices:
                        phase_indices[phase_name] = []
                    phase_indices[phase_name].append(i)

                BUCKET_STEP = 32 # Round all sub-batches to nearest 32
                
                for phase_name, indices in phase_indices.items():
                    # Extract the sub-batch for this phase
                    group_input = policy_context[indices]
                    real_size = group_input.shape[0]
                    
                    # Calculate target bucket size (e.g., 12 -> 32, 45 -> 64)
                    target_size = ((real_size + BUCKET_STEP - 1) // BUCKET_STEP) * BUCKET_STEP
                    
                    pad_amount = target_size - real_size
                    
                    if pad_amount > 0:
                        # Pad the input at the end
                        # F.pad format: (left, right, top, bottom)
                        # We pad dimension 0 (bottom), so: (0, 0, 0, pad_amount)
                        padded_input = F.pad(group_input, (0, 0, 0, pad_amount))
                        
                        # Pass through the layer (MPS compiles ONE graph for size Target_Size)
                        padded_logits = self.policy_heads[phase_name](padded_input)
                        
                        # Slice off the padding so gradients are correct
                        group_logits = padded_logits[:real_size]
                    else:
                        group_logits = self.policy_heads[phase_name](group_input)

                    policy_logits[indices, :group_logits.shape[1]] = group_logits


        # --- Auxiliary Ship Hull Head ---
        scalar_expanded = scalar_final_state.unsqueeze(1).expand(-1, N, -1)
        hull_head_input = torch.cat([all_ship_final_states, scalar_expanded], dim=2)
        predicted_hull = self.hull_head(hull_head_input).squeeze(-1)

        # --- Auxiliary Game Length Head ---
        predicted_game_length = self.game_length_head(scalar_final_state)


        # --- 5. Return all outputs ---
        outputs = {
            "policy_logits": policy_logits,
            "value": value,
            "predicted_hull": predicted_hull,
            "predicted_game_length": predicted_game_length
        }
        return outputs


    def _init_weights(self, module):
        """
        Standard AlphaZero/MuZero style initialization.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Kaiming Normal for ReLU networks
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.BatchNorm2d):
            # Standard BN initialization
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # Transformer specific (PyTorch defaults are usually fine, but being explicit helps)
        elif isinstance(module, nn.MultiheadAttention):
            # Projections are Linear, so they might be caught above, 
            # but usually we want Xavier for attention components
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)

    def initialize_parameters(self):
        """
        Apply initialization to all sub-modules.
        """
        # 1. Apply recursive initialization
        self.apply(self._init_weights)
        
        # 2. Special handling for GlobalPoolingBias (Keep your logic)
        # Your class already does this in __init__, but re-applying ensures consistency
        for m in self.modules():
            if isinstance(m, GlobalPoolingBias):
                nn.init.zeros_(m.linear.weight)
                nn.init.zeros_(m.linear.bias)

        # 3. "Zero Gamma" Trick for ResNets
        # Initialize the last BN in each ResBlock to zero so the block starts as Identity
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.bn2.weight)  # The last BN in your ResBlock
                
        # 4. Special handling for Heads (Optional but recommended)
        # Start with lower variance to prevent confident random outputs
        for phase_head in self.policy_heads.values():
            # Last layer of the policy MLP
            last_linear = phase_head[-1] #type: ignore
            nn.init.normal_(last_linear.weight, mean=0, std=0.01)
            nn.init.zeros_(last_linear.bias)
            
        # Value head last layer
        nn.init.normal_(self.value_head[-2].weight, mean=0, std=0.01) # -2 because Tanh is last #type: ignore
        nn.init.zeros_(self.value_head[-2].bias) #type: ignore