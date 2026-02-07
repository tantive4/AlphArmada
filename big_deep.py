import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

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
    
class UnifiedPointerHead(nn.Module):
    """
    Shared Pointer Logic for both Ships and Defense Tokens.
    Projects different entity types into a common 'Key' space for attention.
    DOES NOT handle 'Pass' (that is now a static action).
    """
    def __init__(self, context_dim, ship_dim, token_dim, attn_dim=64):
        super(UnifiedPointerHead, self).__init__()
        
        # 1. Project Context (Query) -> Common Space
        self.query_proj = nn.Linear(context_dim, attn_dim)
        
        # 2. Project Ships (Key A) -> Common Space
        self.ship_proj = nn.Linear(ship_dim, attn_dim)
        
        # 3. Project Tokens (Key B) -> Common Space
        self.token_proj = nn.Linear(token_dim, attn_dim)
        
        self.scale = attn_dim ** -0.5

    def forward(self, context, candidates, candidate_type='ship'):
        """
        Args:
            context: [B, Context_Dim]
            candidates: 
                - If type='ship': [B, N_Ships, Ship_Dim]
                - If type='token': [B, N_Tokens, Token_Dim]
            candidate_type: 'ship' or 'token'
        Returns:
            scores: [B, N_Candidates] (Logits, unmasked)
        """
        # Q: [B, 1, Attn_Dim]
        Q = self.query_proj(context).unsqueeze(1)
        
        # K: [B, N, Attn_Dim]
        if candidate_type == 'ship':
            K = self.ship_proj(candidates)
        elif candidate_type == 'token':
            K = self.token_proj(candidates)
        else:
            raise ValueError(f"Unknown candidate type: {candidate_type}")
            
        # Attention Scores: [B, 1, Attn_Dim] @ [B, Attn_Dim, N] -> [B, 1, N]
        scores = torch.matmul(Q, K.transpose(1, 2)).squeeze(1)
        scores = scores * self.scale
        
        return scores
    
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
        self.token_feat_size = Config.DEF_TOKEN_FEATURE_SIZE
        
        # Main Embedding Dimension (d_model)
        self.embed_dim = 256 
        self.token_embed_dim = 16
        self.nhead = 8
        self.policy_input_dim = self.embed_dim * 2 # Active Ship + Global State

        # Coordinate Fourier Embedding
        self.num_freqs = 10
        self.coord_embed_dim = 3 * self.num_freqs * 2 # X/Y/θ * freq * sin/cos

        # --- 1. Embeddings ---
        # Scalar Token Encoder (The [CLS] Token)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_feat_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim),
        )

        # Defense Token Encoder
        # Processes raw token features [B, N, 4, 8] -> [B, N, 4, 64]
        self.token_encoder = nn.Sequential(
            nn.Linear(self.token_feat_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.token_embed_dim)
        )

        # --- 2. Ship Embedding (Fusion via Summation) ---
        # Input: Raw Ship Features + (Sum of Token Embeddings)
        # Size: Raw_Ship_Feats + 16
        self.ship_input_dim = self.ship_feat_size + self.token_embed_dim + self.coord_embed_dim
        
        self.ship_embedding = nn.Linear(self.ship_input_dim, self.embed_dim)

        # --- 2. Attention Bias Parameters ---
        # The Scalar Token has no geometry. We learn its relationship to ships.
        # Shape: [Heads, 1, N] and [Heads, N, 1]
        self.relation_bias_net = nn.Sequential(
            nn.Linear(20, 32),
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
        self.transformer_block1 = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)


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
        self.bit_mask = cast(torch.Tensor, self.bit_mask)
        
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
        self.transformer_block2 = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)


        # --- 6. Output Heads ---
        
        # Value Head: Uses Scalar Token (The "Game State")
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # A. Shared Pointer Head (For Type 1 & Type 2)
        # Note: 'ship_dim' here is the final transformer output (256)
        # 'token_dim' is the small embedding (64)
        self.unified_pointer_head = UnifiedPointerHead(
            context_dim=self.policy_input_dim,
            ship_dim=self.embed_dim,
            token_dim=self.token_embed_dim,
            attn_dim=64
        )

        # B. Static Heads (For Type 1 Pass, Type 2 Static, Type 3 All)
        self.static_heads = nn.ModuleDict()
        
        # Define Phase Groups
        self.ship_pointer_phases = {Phase.SHIP_ACTIVATE, Phase.SHIP_CHOOSE_TARGET_SHIP}
        self.token_pointer_phases = {Phase.ATTACK_RESOLVE_EFFECTS, Phase.ATTACK_SPEND_DEFENSE_TOKENS}

        for phase in Phase:
            actions = self.action_manager.get_action_map(phase)
            if not actions: continue

            # Determine Output Size for the Static Head
            if phase in self.ship_pointer_phases:
                # Type 1: Only "Pass" is static
                static_out_dim = 1 
                
            elif phase in self.token_pointer_phases:
                # Type 2: All actions EXCEPT the token slots (0..3)
                # We assume actions 0..3 are tokens, 4+ are static
                static_out_dim = len(actions) - Config.MAX_DEFENSE_TOKENS
                
            else:
                # Type 3: All actions are static
                static_out_dim = len(actions)

            # Create the Static MLP
            self.static_heads[phase.name] = nn.Sequential(
                nn.Linear(self.policy_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, static_out_dim)
            )


        self.active_phases = sorted(
            [p for p in Phase if p.name in self.static_heads], 
            key=lambda x: x.value
        )
        self.max_static_action_space = 0
        for phase in self.active_phases:
            head = cast(nn.Sequential, self.static_heads[phase.name])
            # The last linear layer determines output size
            out_dim = cast(int, head[-1].out_features)
            self.max_static_action_space = max(self.max_static_action_space, out_dim)


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

        # Auxiliary: Win Rate
        # Input: Scalar Token (256)
        self.win_prob_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        



    def compile_fast_policy(self):
        """
        Compiles the individual ModuleDict static heads into stacked tensors for 
        efficient batched inference (BMM). Also builds lookup tables for phase types.
        """
        # 1. Build Lookup Tables
        max_phase_val = max(p.value for p in self.active_phases)
        
        # A. Stack Index Lookup (Phase -> Index in W_stack)
        self.phase_lookup = torch.full((max_phase_val + 1,), -1, dtype=torch.long, device=Config.DEVICE)
        
        # B. Phase Type Lookup (Phase -> 0:Std, 1:ShipPtr, 2:TokenPtr)
        # Used for masking in forward pass
        self.phase_type_lookup = torch.full((max_phase_val + 1,), 0, dtype=torch.long, device=Config.DEVICE)
        
        for idx, phase in enumerate(self.active_phases):
            self.phase_lookup[phase.value] = idx
            
            if phase in self.ship_pointer_phases:
                self.phase_type_lookup[phase.value] = 1
            elif phase in self.token_pointer_phases:
                self.phase_type_lookup[phase.value] = 2
            else:
                self.phase_type_lookup[phase.value] = 0

        # 2. Pre-allocate Stacked Weights [Num_Phases, Out, In]
        num_phases = len(self.active_phases)
        
        # Layer 1 (256 -> 256)
        self.w1_stack = torch.zeros(num_phases, 256, self.policy_input_dim, device=Config.DEVICE)
        self.b1_stack = torch.zeros(num_phases, 256, device=Config.DEVICE)
        
        # Layer 2 (256 -> 256)
        self.w2_stack = torch.zeros(num_phases, 256, 256, device=Config.DEVICE)
        self.b2_stack = torch.zeros(num_phases, 256, device=Config.DEVICE)
        
        # Layer 3 (256 -> Max_Static)
        self.w3_stack = torch.zeros(num_phases, self.max_static_action_space, 256, device=Config.DEVICE)
        self.b3_stack = torch.zeros(num_phases, self.max_static_action_space, device=Config.DEVICE)
        
        # Mask to ensure padded logits (indices > real_size) are -inf
        self.static_padding_mask = torch.zeros(num_phases, self.max_static_action_space, device=Config.DEVICE)

        # 3. Copy Weights
        with torch.no_grad():
            for idx, phase in enumerate(self.active_phases):
                head = cast(nn.Sequential, self.static_heads[phase.name])
                
                # Copy L1
                l1 = cast(nn.Linear, head[0])
                self.w1_stack[idx] = l1.weight
                self.b1_stack[idx] = l1.bias
                
                # Copy L2
                l2 = cast(nn.Linear, head[2])
                self.w2_stack[idx] = l2.weight
                self.b2_stack[idx] = l2.bias
                
                # Copy L3 (Handle Padding)
                l3 = cast(nn.Linear, head[4])
                real_out = l3.out_features
                self.w3_stack[idx, :real_out, :] = l3.weight
                self.b3_stack[idx, :real_out] = l3.bias
                
                # Set padding mask: 0 for valid, -inf for padded
                if real_out < self.max_static_action_space:
                    self.static_padding_mask[idx, real_out:] = -1e9

        self.fast_policy_ready = True

    def compute_fourier_features(self, coords):
        """
        coords: [Batch, N, 3] (x, y, θ) normalized 0-1
        Returns: [Batch, N, coord_embed_dim]
        """
        # Create frequencies: 1, 2, 4, 8, 16... (powers of 2 are standard)
        freqs = 2.0 ** torch.arange(self.num_freqs, device=coords.device)
        freqs = freqs * torch.pi # Scale by PI
        
        # Reshape for broadcasting
        # x: [B, N, 2, 1] * freq: [1, 1, 1, F] -> [B, N, 2, F]
        args = coords.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
        
        # Compute Sin/Cos
        # [B, N, 2, F] -> [B, N, 2, F, 2] (last dim is sin/cos)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Flatten to [B, N, 4*F]
        return embeddings.flatten(start_dim=2)


    def forward(self, 
                scalar_input : torch.Tensor, 
                ship_entity_input : torch.Tensor, 
                ship_coord_input : torch.Tensor, 
                ship_token_input : torch.Tensor, 
                spatial_input : torch.Tensor, 
                relation_input : torch.Tensor, 
                active_ship_indices : torch.Tensor, 
                target_ship_indices : torch.Tensor,
                phases : torch.Tensor):
        """
        Args:
            scalar_input: [B, 45]
            ship_entity_input: [B, N, 110]
            ship_coord_input: [B, N, 2] - Normalized (0-1) coordinates for safer indexing
            spatial_input: [B, N, 10, H, W] - Plane 0 is presence, 1-9 are threat geometry
            relation_input: [B, N, N, 20] - Raw 4x4 hull relation matrix (flattened) + 4 geometric information
            active_ship_indices: [B] Tensor. Int index of active ship (0 to N-1). 
                                 Use N (Config.MAX_SHIPS) or -1 for "No Active Ship".
            target_ship_indices: [B] Tensor. Int index of target ship (0 to N-1).
                                 Use N (Config.MAX_SHIPS) or -1 for "No Target Ship".
            phases: [B] - tensor of phases
        """

        batch_size = scalar_input.shape[0]
        N = Config.MAX_SHIPS
        T = Config.MAX_DEFENSE_TOKENS
        device = scalar_input.device


        # === 1. Embedding & Token Construction ===

        # A. Create Scalar Token (The "Game State" / [CLS] Token)
        # [B, 45] -> [B, 1, 256]
        scalar_token = self.scalar_encoder(scalar_input).unsqueeze(1)

        # B. Encode Tokens [B, N, T, 8] -> [B, N, T, 16] -> [B, N, 16]
        token_encoded = self.token_encoder(ship_token_input) 
        token_summed = token_encoded.sum(dim=2)
        
        # C. Coordinate Fourier Embedding
        coord_features = self.compute_fourier_features(ship_coord_input) 

        # D. Fuse with Ship [B, N, Raw] + [B, N, 16] -> [B, N, Raw+16]
        ship_combined_input = torch.cat([ship_entity_input, token_summed, coord_features], dim=2)
        
        # E. Project to Embedding [B, N, 256]
        ship_tokens = self.ship_embedding(ship_combined_input)        

        # F. Combine into Single Sequence
        # Shape: [B, N+1, 256] (Index 0 is Scalar, 1..N are Ships)
        tokens = torch.cat([scalar_token, ship_tokens], dim=1)


        # === 2. Attention Mask & Bias Construction ===

        # A. Padding Mask (Who is real?)
        # [B, N] - True if ship exists
        valid_ship_mask = (ship_entity_input.abs().sum(dim=2) > 0)
        
        # [B, N+1] - Initialize with False (Keep all)
        # In PyTorch MultiheadAttention, True values are IGNORED.
        src_key_padding_mask = torch.zeros(batch_size, N + 1, dtype=scalar_input.dtype, device=device)
        
        # Mask out invalid ships (indices 1 to N). Index 0 (Scalar) is always kept.
        src_key_padding_mask[:, 1:] = src_key_padding_mask[:, 1:].masked_fill(~valid_ship_mask, float('-inf'))

        # B. Attention Bias (Who relates to whom?)
        # 1. Process Geometric Relations (Ship-to-Ship)
        # [B, N, N, 20] -> [B, N, N, Heads] -> [B, Heads, N, N]
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
        grid_coords = ship_coord_input[:, :, :2].unsqueeze(1) * 2 - 1 
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
        if not self.training and getattr(self, 'fast_policy_ready', False):
            # =========================================================
            # FAST PATH: Unified Batch (BMM + Masked Pointer)
            # =========================================================
            
            # 1. Phase Type Masks
            # 0=Std, 1=ShipPtr, 2=TokenPtr
            phase_types = self.phase_type_lookup[phases] # [B]
            
            is_ship_ptr = (phase_types == 1).view(batch_size, 1, 1) # [B, 1, 1]
            is_token_ptr = (phase_types == 2).view(batch_size, 1, 1) # [B, 1, 1]

            # 2. Unified Pointer Candidates
            # We need to construct a [B, N, Candidate_Dim] tensor.
            # Only N candidates are allowed. Tokens (size T) are padded to N.
            
            # A. Token Candidates (Gathered from Active Ship)
            # gather indices: [B, 1, 1, 1] -> expand to [B, 1, T, 16]
            # using 'token_encoded' from step 1A
            # padding Input: [B, N, T, 16] -> Output: [B, N+1, T, 16]
            token_encoded_padded = F.pad(token_encoded, (0, 0, 0, 0, 0, 1))
            token_gather_idx = target_ship_indices.view(batch_size, 1, 1, 1).expand(-1, 1, T, self.token_embed_dim)
            active_tokens_raw = torch.gather(token_encoded_padded, 1, token_gather_idx).squeeze(1) # [B, T, 16]
            
            # B. Project Candidates to Attention Space (Keys)
            # Ship Keys: [B, N, 256] -> [B, N, 64]
            # Token Keys: [B, T, 16] -> [B, T, 64]
            # Note: We call the projection layers manually to avoid the 'candidate_type' check in forward
            keys_ship = self.unified_pointer_head.ship_proj(all_ship_final_states)
            keys_token_raw = self.unified_pointer_head.token_proj(active_tokens_raw)
            keys_token = F.pad(keys_token_raw, (0, 0, 0, N - T)) # Pad T -> N [B, N, 64]

            # C. Mix Keys based on Phase
            # If Ship Phase: use Ship Keys. If Token Phase: use Token Keys. Else: 0.
            candidate_keys = (keys_ship * is_ship_ptr) + (keys_token * is_token_ptr)

            # D. Pointer Attention (Query vs Mixed Keys)
            # [B, 1, 64] @ [B, 64, N] -> [B, N]
            # We call the internal projection manually to share the logic
            query = self.unified_pointer_head.query_proj(policy_context).unsqueeze(1)
            pointer_logits = torch.matmul(query, candidate_keys.transpose(1, 2)).squeeze(1)
            pointer_logits = pointer_logits * self.unified_pointer_head.scale

            # E. Pointer Masking
            # Mask out:
            # 1. Invalid Ships (if Ship Phase)
            # 2. Invalid Tokens (if Token Phase)
            # 3. Everything (if Standard Phase)
            
            valid_ship_mask = (ship_entity_input.abs().sum(dim=2) > 0) # [B, N]
            valid_token_mask_raw = (active_tokens_raw.abs().sum(dim=2) > 0) # [B, T]
            valid_token_mask = F.pad(valid_token_mask_raw, (0, N-T)) # [B, N]

            seq_indices = torch.arange(N, device=device).unsqueeze(0).expand(batch_size, -1)
            is_active_ship_mask = (seq_indices == active_ship_indices.unsqueeze(1))
            
            # Combine validity with phase type
            final_pointer_mask = (valid_ship_mask & is_ship_ptr.squeeze(2)) | \
                                 (valid_token_mask & is_token_ptr.squeeze(2))
            
            final_pointer_mask = final_pointer_mask & (~is_active_ship_mask)
            
            # Apply -inf mask
            pointer_logits = pointer_logits.masked_fill(~final_pointer_mask, -1e9)

            # 3. Unified Static Heads (BMM)
            stack_indices = self.phase_lookup[phases]
            
            # Layer 1
            w1, b1 = self.w1_stack[stack_indices], self.b1_stack[stack_indices]
            x = policy_context.unsqueeze(2) # [B, 512, 1]
            x = F.relu(torch.bmm(w1, x).squeeze(2) + b1).unsqueeze(2)
            
            # Layer 2
            w2, b2 = self.w2_stack[stack_indices], self.b2_stack[stack_indices]
            x = F.relu(torch.bmm(w2, x).squeeze(2) + b2).unsqueeze(2)
            
            # Layer 3
            w3, b3 = self.w3_stack[stack_indices], self.b3_stack[stack_indices]
            static_logits = torch.bmm(w3, x).squeeze(2) + b3 # [B, Max_Static]
            
            # Apply Pre-calculated Padding Mask (for varying output sizes)
            pad_mask = self.static_padding_mask[stack_indices]
            static_logits = static_logits + pad_mask

            # 4. Concatenate
            policy_logits = torch.cat([pointer_logits, static_logits], dim=1)

        else:
            # =========================================================
            # SLOW PATH: Training / Legacy (Loop-based)
            # =========================================================
            policy_logits = torch.zeros(batch_size, N + self.max_static_action_space, device=device)
            # Initialize with large negative for softmax safety
            policy_logits.fill_(-1e9) 
            
            # Group by Phase Name
            phase_indices = {}
            for i, p in enumerate(phases):
                p_name = Phase(p.item()).name
                if p_name not in phase_indices: phase_indices[p_name] = []
                phase_indices[p_name].append(i)

            for phase_name, indices in phase_indices.items():
                indices_tensor = torch.tensor(indices, device=device)
                group_context = policy_context[indices_tensor]
                phase_enum = Phase[phase_name]
                
                # 1. Static Head Calculation
                group_static_logits = self.static_heads[phase_name](group_context)
                
                # Place static logits at the END of the vector
                # (Pointer is 0..N, Static is N..End)
                # We need to map [B_sub, Static_Out] -> [B_sub, Total_Out]
                static_len = group_static_logits.shape[1]
                policy_logits[indices_tensor, N : N+static_len] = group_static_logits

                # 2. Pointer Head Calculation (If applicable)
                if phase_enum in self.ship_pointer_phases:
                    # Type 1: Ship Pointer
                    # Gather ships for this group
                    group_ships = all_ship_final_states[indices_tensor]
                    
                    ptr_logits = self.unified_pointer_head(
                        group_context, group_ships, candidate_type='ship'
                    )
                    
                    # Apply Mask
                    group_mask = (ship_entity_input[indices_tensor].abs().sum(dim=2) > 0)
                    ptr_logits = ptr_logits.masked_fill(~group_mask, -1e9)
                    
                    policy_logits[indices_tensor, :N] = ptr_logits

                elif phase_enum in self.token_pointer_phases:
                    # Type 2: Token Pointer
                    # Gather tokens for this group
                    # indices_tensor -> [Sub_B]
                    # active_ship_indices -> [B] -> [Sub_B]
                    group_active_ships = active_ship_indices[indices_tensor]
                    
                    # Gather tokens
                    sub_b = len(indices)
                    gather_idx = group_active_ships.view(sub_b, 1, 1, 1).expand(-1, 1, T, self.token_embed_dim)
                    group_tokens_encoded = token_encoded[indices_tensor] # [Sub_B, N, T, 16]
                    group_active_tokens = torch.gather(group_tokens_encoded, 1, gather_idx).squeeze(1)
                    
                    ptr_logits = self.unified_pointer_head(
                        group_context, group_active_tokens, candidate_type='token'
                    )
                    
                    # Apply Mask (Pad T to N logic handled by putting it in 0..T slice)
                    # We need to explicitly pad the result to N if T < N
                    if T < N:
                         ptr_logits = F.pad(ptr_logits, (0, N-T), value=-1e9)

                    # Validity Mask
                    group_token_mask = (group_active_tokens.abs().sum(dim=2) > 0)
                    # Pad mask to N
                    group_token_mask = F.pad(group_token_mask, (0, N-T))
                    
                    ptr_logits = ptr_logits.masked_fill(~group_token_mask, -1e9)

                    policy_logits[indices_tensor, :N] = ptr_logits


        # --- Auxiliary Ship Hull Head ---
        scalar_expanded = scalar_final_state.unsqueeze(1).expand(-1, N, -1)
        hull_head_input = torch.cat([all_ship_final_states, scalar_expanded], dim=2)
        predicted_hull = self.hull_head(hull_head_input).squeeze(-1)

        # --- Auxiliary Game Length Head ---
        predicted_game_length = self.game_length_head(scalar_final_state)

        # --- Auxiliary Win Rate Head ---
        win_prob = self.win_prob_head(scalar_final_state)


        # --- 5. Return all outputs ---
        outputs = {
            "policy_logits": policy_logits,
            "value": value,
            "predicted_win_prob": win_prob,
            "predicted_hull": predicted_hull,
            "predicted_game_length": predicted_game_length
        }
        return outputs
