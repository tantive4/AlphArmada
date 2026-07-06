import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

# Import the configuration constants from your encoder to ensure the model's
# input shapes match the encoder's output shapes perfectly.
from learning.params.configs import Config
from action_manager import ActionManager
from armada_game.helpers.action_phase import Phase
from armada_game.helpers.enum_class import *

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
    Order: BN -> GELU -> Conv -> BN -> GELU -> Conv
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
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)

        out = F.gelu(self.bn2(out))
        out = self.conv2(out)

        # Global Pooling Bias applied within the branch
        if self.use_global_pooling:
            out = self.gpb(out)

        # Addition (Residual Connection)
        return x + out

class PointerAttentionHead(nn.Module):
    """
    Dot-product pointer over one candidate type.
    """
    def __init__(self, context_dim, candidate_dim, attn_dim=64):
        super(PointerAttentionHead, self).__init__()
        self.query_proj = nn.Linear(context_dim, attn_dim)
        self.key_proj = nn.Linear(candidate_dim, attn_dim)
        self.scale = attn_dim ** -0.5

    def forward(self, context, candidates):
        query = self.query_proj(context).unsqueeze(1)
        keys = self.key_proj(candidates)
        scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        return scores * self.scale

# --- Main Network Architecture ---

class BigDeep(nn.Module):
    """
    Transformer-Centric "Sandwich" Architecture.
    Sequence: [Scalar_Token, Ship_1, ..., Ship_N]
    Flow: Embed -> Block1 -> Spatial Sandwich -> Block2 -> Heads

    BMM-only policy-head variant. Static policy heads are stored as stacked
    trainable parameters and selected by phase for both training and inference.
    """
    def __init__(self, action_manager: ActionManager):
        super(BigDeep, self).__init__()
        self.action_manager = action_manager

        # --- Constants & Configuration ---
        self.ship_feat_size = Config.SHIP_ENTITY_FEATURE_SIZE
        self.scalar_feat_size = Config.SCALAR_FEATURE_SIZE
        self.token_feat_size = Config.DEF_TOKEN_FEATURE_SIZE
        self.ship_action_offset = 0
        self.ship_pointer_action_size = Config.MAX_SHIPS
        self.token_action_offset = self.ship_action_offset + self.ship_pointer_action_size
        self.token_pointer_action_size = Config.MAX_DEFENSE_TOKENS
        self.static_action_offset = self.token_action_offset + self.token_pointer_action_size

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
            nn.GELU(),
            nn.Linear(128, self.embed_dim),
        )

        # Defense Token Encoder
        # Processes raw token features [B, N, 4, 8] -> [B, N, 4, 64]
        self.token_encoder = nn.Sequential(
            nn.Linear(self.token_feat_size, 32),
            nn.GELU(),
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
            nn.GELU(),
            nn.Linear(32, self.nhead)
        )
        self.bias_scale = 10.0

        # Shape: [Heads, 1, 1]
        self.scalar_bias_row = nn.Parameter(torch.randn(self.nhead, 1, 1) * 0.1)
        self.scalar_bias_col = nn.Parameter(torch.randn(self.nhead, 1, 1) * 0.1)
        self.scalar_self_bias = nn.Parameter(torch.randn(self.nhead, 1, 1) * 0.1)

        # --- DEFINE NORMS ---
        # Add these new layers
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm_fusion = nn.LayerNorm(self.embed_dim)
        self.norm_policy = nn.LayerNorm(self.policy_input_dim)


        # --- 3. Transformer Block 1 (Geometry Aware) ---
        # "Reasoning about immediate geometric relations"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.nhead,
            dim_feedforward=1024,
            batch_first=False,
            norm_first=True,
            activation='gelu'
        )
        self.transformer_block1 = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)


        # --- 4. Spatial Sandwich Components ---

        # Projectors (Transformer -> Spatial Map)
        self.presence_channels = 32
        self.presence_projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.presence_channels)
        )

        self.threat_channels = 4
        self.num_threat_planes = 9
        self.threat_projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.threat_channels * self.num_threat_planes)
        )

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
            nn.GELU()
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
        self.transformer_block2 = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)


        # --- 6. Output Heads ---

        # Value Head: Uses Scalar Token (The "Game State")
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.ship_pointer_head = PointerAttentionHead(
            context_dim=self.policy_input_dim,
            candidate_dim=self.embed_dim,
            attn_dim=64
        )
        self.token_pointer_head = PointerAttentionHead(
            context_dim=self.policy_input_dim,
            candidate_dim=self.token_embed_dim,
            attn_dim=64
        )

        # Define Phase Groups
        self.ship_pointer_phases = {Phase.SHIP_ACTIVATE, Phase.SHIP_CHOOSE_TARGET_SHIP}
        self.token_pointer_phases = {Phase.ATTACK_RESOLVE_EFFECTS, Phase.ATTACK_SPEND_DEFENSE_TOKENS}

        self.active_phases = []
        static_out_dims = []
        max_phase_val = max(p.value for p in Phase)
        phase_lookup = torch.full((max_phase_val + 1,), -1, dtype=torch.long)
        phase_type_lookup = torch.full((max_phase_val + 1,), 0, dtype=torch.long)

        for phase in Phase:
            actions = self.action_manager.get_action_map(phase)
            if not actions: continue
            phase_lookup[phase.value] = len(self.active_phases)
            self.active_phases.append(phase)

            # Determine Output Size for the Static Head
            if phase in self.ship_pointer_phases:
                static_out_dim = len(actions) - Config.MAX_SHIPS
                phase_type_lookup[phase.value] = 1

            elif phase in self.token_pointer_phases:
                # Type 2: All actions EXCEPT the token slots (0..3)
                # We assume actions 0..3 are tokens, 4+ are static
                static_out_dim = len(actions) - Config.MAX_DEFENSE_TOKENS
                phase_type_lookup[phase.value] = 2

            else:
                # Type 3: All actions are static
                static_out_dim = len(actions)

            static_out_dims.append(static_out_dim)

        self.max_static_action_space = max(static_out_dims)
        self.max_action_space = self.static_action_offset + self.max_static_action_space
        self.register_buffer('phase_lookup', phase_lookup)
        self.register_buffer('phase_type_lookup', phase_type_lookup)

        static_padding_mask = torch.full((len(self.active_phases), self.max_static_action_space), -1e9)
        for idx, real_out in enumerate(static_out_dims):
            if real_out > 0:
                static_padding_mask[idx, :real_out] = 0.0
        self.register_buffer('static_padding_mask', static_padding_mask)

        # Stacked trainable static policy MLP:
        # [phase, out, in] tensors are selected per sample and used with BMM.
        num_phases = len(self.active_phases)
        self.w1_stack = nn.Parameter(torch.empty(num_phases, 256, self.policy_input_dim))
        self.b1_stack = nn.Parameter(torch.empty(num_phases, 256))
        self.w2_stack = nn.Parameter(torch.empty(num_phases, 256, 256))
        self.b2_stack = nn.Parameter(torch.empty(num_phases, 256))
        self.w3_stack = nn.Parameter(torch.empty(num_phases, self.max_static_action_space, 256))
        self.b3_stack = nn.Parameter(torch.empty(num_phases, self.max_static_action_space))
        self._reset_policy_parameters()


        # Auxiliary: Hull Prediction
        # Input: Ship Token (256) + Scalar Token (256)
        self.hull_head = nn.Sequential(
            nn.Linear(self.policy_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Auxiliary: Game Length
        # Input: Scalar Token (256)
        self.game_length_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )

        # Auxiliary: Win Rate
        # Input: Scalar Token (256)
        self.raw_point_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )





    def _reset_policy_parameters(self):
        """Initialize each stacked static policy layer like nn.Linear."""
        for weight, bias in (
            (self.w1_stack, self.b1_stack),
            (self.w2_stack, self.b2_stack),
            (self.w3_stack, self.b3_stack),
        ):
            for idx in range(weight.shape[0]):
                nn.init.kaiming_uniform_(weight[idx], a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight[idx])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias[idx], -bound, bound)

    @staticmethod
    def _normalize_ship_indices(indices, max_ships):
        """Map invalid ship ids, including -1, to the padded sentinel slot."""
        indices = indices.long()
        sentinel = torch.full_like(indices, max_ships)
        return torch.where((indices >= 0) & (indices < max_ships), indices, sentinel)

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

    def _build_token_sequence(self, scalar_input, ship_entity_input, ship_coord_input, ship_token_input):
        # === 1. Embedding & Token Construction ===
        scalar_token = self.scalar_encoder(scalar_input).unsqueeze(1)

        token_valid_mask = ship_token_input.abs().sum(dim=3) > 0
        token_encoded = self.token_encoder(ship_token_input)
        token_summed = (token_encoded * token_valid_mask.unsqueeze(-1).float()).sum(dim=2)

        coord_features = self.compute_fourier_features(ship_coord_input)
        ship_combined_input = torch.cat([ship_entity_input, token_summed, coord_features], dim=2)
        ship_tokens = self.ship_embedding(ship_combined_input)
        tokens = torch.cat([scalar_token, ship_tokens], dim=1)

        return scalar_token, token_encoded, token_valid_mask, tokens

    def _build_attention_inputs(self, scalar_input, ship_entity_input, relation_input):
        # === 2. Attention Mask & Bias Construction ===
        batch_size = scalar_input.shape[0]
        N = Config.MAX_SHIPS
        device = scalar_input.device

        valid_ship_mask = ship_entity_input.abs().sum(dim=2) > 0

        src_key_padding_mask = torch.zeros(batch_size, N + 1, dtype=scalar_input.dtype, device=device)
        src_key_padding_mask[:, 1:] = src_key_padding_mask[:, 1:].masked_fill(~valid_ship_mask, float('-inf'))

        geom_bias = self.relation_bias_net(relation_input)
        geom_bias = torch.tanh(geom_bias) * self.bias_scale
        geom_bias = geom_bias.permute(0, 3, 1, 2)

        full_bias = torch.zeros(batch_size, self.nhead, N + 1, N + 1, device=device)
        full_bias[:, :, 1:, 1:] = geom_bias

        scaled_row_bias = self.scalar_bias_row * self.bias_scale
        scaled_col_bias = self.scalar_bias_col * self.bias_scale
        scaled_self_bias = self.scalar_self_bias * self.bias_scale

        full_bias[:, :, 0:1, 1:] = scaled_row_bias.unsqueeze(0).expand(batch_size, -1, -1, N)
        full_bias[:, :, 1:, 0:1] = scaled_col_bias.unsqueeze(0).expand(batch_size, -1, N, -1)
        full_bias[:, :, 0, 0] = scaled_self_bias.squeeze().unsqueeze(0).expand(batch_size, -1)

        attn_bias = full_bias.reshape(batch_size * self.nhead, N + 1, N + 1)
        return valid_ship_mask, src_key_padding_mask, attn_bias

    def _run_geometric_transformer(self, tokens, attn_bias, src_key_padding_mask):
        # === 3. Transformer Block 1 (Geometric Reasoning) ===
        tokens = tokens.permute(1, 0, 2)
        tokens_l1 = self.transformer_block1(tokens, mask=attn_bias, src_key_padding_mask=src_key_padding_mask)
        tokens_l1 = tokens_l1.permute(1, 0, 2)
        return self.norm1(tokens_l1)

    def _run_spatial_sandwich(self, tokens_l1, scalar_token, ship_coord_input, spatial_input):
        # === 4. Spatial Sandwich (Scatter -> ResNet -> Gather) ===
        batch_size = tokens_l1.shape[0]
        N = Config.MAX_SHIPS
        device = tokens_l1.device

        ships_l1 = tokens_l1[:, 1:]

        unpacked_spatial = (spatial_input.unsqueeze(-1) & self.bit_mask) > 0
        unpacked_spatial = unpacked_spatial.flatten(start_dim=-2).float()

        presence_vals = self.presence_projector(ships_l1)
        threat_vals = self.threat_projector(ships_l1).view(batch_size, N, self.threat_channels, self.num_threat_planes)

        presence_map = torch.einsum('bnc, bnhw -> bchw', presence_vals, unpacked_spatial[:, :, 0])
        threat_map = torch.einsum('bncg, bnghw -> bchw', threat_vals, unpacked_spatial[:, :, 1:])
        spatial_combined = torch.cat([presence_map, threat_map], dim=1)

        x = self.spatial_head_conv(spatial_combined)
        scalar_embed_raw = scalar_token.squeeze(1)
        global_bias = self.spatial_global_bias(scalar_embed_raw).unsqueeze(-1).unsqueeze(-1)
        x = x + global_bias
        x = self.spatial_trunk(x)
        spatial_features_map = self.spatial_tail(x)

        grid_coords = ship_coord_input[:, :, :2].unsqueeze(1) * 2 - 1
        grid_coords = grid_coords.clamp(-1, 1)

        if self.training and device.type == 'mps':
            gathered_spatial = F.grid_sample(
                spatial_features_map.cpu(), grid_coords.cpu(), align_corners=False, padding_mode='zeros'
            ).to(device)
        else:
            gathered_spatial = F.grid_sample(
                spatial_features_map, grid_coords, align_corners=False, padding_mode='zeros'
            )
        ship_spatial_ctx = gathered_spatial.squeeze(2).transpose(1, 2)

        sp_avg = F.adaptive_avg_pool2d(spatial_features_map, 1).flatten(1)
        sp_max = F.adaptive_max_pool2d(spatial_features_map, 1).flatten(1)
        scalar_spatial_raw = torch.cat([sp_avg, sp_max], dim=1)
        scalar_spatial_ctx = self.scalar_spatial_adapter(scalar_spatial_raw).unsqueeze(1)

        spatial_ctx = torch.cat([scalar_spatial_ctx, ship_spatial_ctx], dim=1)
        fused_input = torch.cat([tokens_l1, spatial_ctx], dim=2)
        tokens_l2_input = self.sandwich_fusion(fused_input)
        return self.norm_fusion(tokens_l2_input)

    def _run_tactical_transformer(self, tokens_l2_input, attn_bias, src_key_padding_mask):
        # === 5. Transformer Block 2 (Tactical Reasoning) ===
        tokens_l2_input = tokens_l2_input.permute(1, 0, 2)
        final_tokens = self.transformer_block2(tokens_l2_input, mask=attn_bias, src_key_padding_mask=src_key_padding_mask)
        final_tokens = final_tokens.permute(1, 0, 2)
        return self.norm2(final_tokens)

    def _prepare_head_inputs(self, final_tokens, valid_ship_mask, active_ship_indices):
        # === 6. Head Preparation ===
        batch_size = final_tokens.shape[0]
        device = final_tokens.device

        scalar_final_state = final_tokens[:, 0]
        all_ship_final_states = final_tokens[:, 1:]
        all_ship_final_states = all_ship_final_states * valid_ship_mask.unsqueeze(-1).float()

        zero_ship = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        lookup_ships = torch.cat([all_ship_final_states, zero_ship], dim=1)
        gather_indices = active_ship_indices.view(batch_size, 1, 1).expand(-1, -1, self.embed_dim)
        active_ship_state = torch.gather(lookup_ships, 1, gather_indices).squeeze(1)

        policy_context = torch.cat([active_ship_state, scalar_final_state], dim=1)
        policy_context = self.norm_policy(policy_context)

        return scalar_final_state, all_ship_final_states, policy_context

    def _compute_output_heads(
        self,
        scalar_final_state,
        all_ship_final_states,
        policy_context,
        token_encoded,
        token_valid_mask,
        valid_ship_mask,
        active_ship_indices,
        target_ship_indices,
        phases,
    ):
        # === 7. Output Heads ===
        batch_size = scalar_final_state.shape[0]
        N = Config.MAX_SHIPS
        T = Config.MAX_DEFENSE_TOKENS
        device = scalar_final_state.device

        value = self.value_head(scalar_final_state)

        phase_types = self.phase_type_lookup[phases]
        is_ship_ptr = (phase_types == 1).view(batch_size, 1)
        is_token_ptr = (phase_types == 2).view(batch_size, 1)

        token_encoded_padded = F.pad(token_encoded, (0, 0, 0, 0, 0, 1))
        token_gather_idx = target_ship_indices.view(batch_size, 1, 1, 1).expand(-1, 1, T, self.token_embed_dim)
        target_tokens_raw = torch.gather(token_encoded_padded, 1, token_gather_idx).squeeze(1)
        token_valid_padded = torch.cat(
            [token_valid_mask, torch.zeros(batch_size, 1, T, dtype=torch.bool, device=device)],
            dim=1,
        )
        target_token_valid = torch.gather(
            token_valid_padded,
            1,
            target_ship_indices.view(batch_size, 1, 1).expand(-1, 1, T),
        ).squeeze(1)

        ship_logits = self.ship_pointer_head(policy_context, all_ship_final_states)
        token_logits = self.token_pointer_head(policy_context, target_tokens_raw)

        seq_indices = torch.arange(N, device=device).unsqueeze(0).expand(batch_size, -1)
        is_active_ship_mask = seq_indices == active_ship_indices.unsqueeze(1)
        ship_mask = is_ship_ptr & valid_ship_mask & (~is_active_ship_mask)
        ship_logits = ship_logits.masked_fill(~ship_mask, -1e9)

        token_mask = is_token_ptr & target_token_valid
        token_logits = token_logits.masked_fill(~token_mask, -1e9)

        stack_indices = self.phase_lookup[phases]
        w1, b1 = self.w1_stack[stack_indices], self.b1_stack[stack_indices]
        x = policy_context.unsqueeze(2)
        x = F.gelu(torch.bmm(w1, x).squeeze(2) + b1).unsqueeze(2)

        w2, b2 = self.w2_stack[stack_indices], self.b2_stack[stack_indices]
        x = F.gelu(torch.bmm(w2, x).squeeze(2) + b2).unsqueeze(2)

        w3, b3 = self.w3_stack[stack_indices], self.b3_stack[stack_indices]
        static_logits = torch.bmm(w3, x).squeeze(2) + b3
        static_logits = static_logits + self.static_padding_mask[stack_indices]

        policy_logits = torch.cat([ship_logits, token_logits, static_logits], dim=1)

        scalar_expanded = scalar_final_state.unsqueeze(1).expand(-1, N, -1)
        hull_head_input = torch.cat([all_ship_final_states, scalar_expanded], dim=2)
        predicted_hull = self.hull_head(hull_head_input).squeeze(-1)
        predicted_game_length = self.game_length_head(scalar_final_state)
        raw_point = self.raw_point_head(scalar_final_state)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "predicted_raw_point": raw_point,
            "predicted_hull": predicted_hull,
            "predicted_game_length": predicted_game_length
        }


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

        N = Config.MAX_SHIPS
        active_ship_indices = self._normalize_ship_indices(active_ship_indices, N)
        target_ship_indices = self._normalize_ship_indices(target_ship_indices, N)

        scalar_token, token_encoded, token_valid_mask, tokens = self._build_token_sequence(
            scalar_input,
            ship_entity_input,
            ship_coord_input,
            ship_token_input,
        )
        valid_ship_mask, src_key_padding_mask, attn_bias = self._build_attention_inputs(
            scalar_input,
            ship_entity_input,
            relation_input,
        )
        tokens_l1 = self._run_geometric_transformer(tokens, attn_bias, src_key_padding_mask)
        tokens_l2_input = self._run_spatial_sandwich(
            tokens_l1,
            scalar_token,
            ship_coord_input,
            spatial_input,
        )
        final_tokens = self._run_tactical_transformer(tokens_l2_input, attn_bias, src_key_padding_mask)
        scalar_final_state, all_ship_final_states, policy_context = self._prepare_head_inputs(
            final_tokens,
            valid_ship_mask,
            active_ship_indices,
        )
        return self._compute_output_heads(
            scalar_final_state,
            all_ship_final_states,
            policy_context,
            token_encoded,
            token_valid_mask,
            valid_ship_mask,
            active_ship_indices,
            target_ship_indices,
            phases,
        )


def load_recent_model()-> tuple[BigDeep, int]:
    """
    Loads the BigDeep model from the latest checkpoint if available.
    If no checkpoint exists, initializes a new model and saves the initial state.
    """
    model = BigDeep(ActionManager()).to(Config.DEVICE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Find all checkpoint files
    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('model_iter_') and f.endswith('.pth')]

    if checkpoints:
        # Find the checkpoint with the highest iteration number
        latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))

        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint_file)
        print(f"[LOAD MODEL] {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        current_iter = int(latest_checkpoint_file.split('_')[-1].split('.')[0])

    else:
        init_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_iter_000.pth")
        torch.save(model.state_dict(), init_checkpoint_path)
        print(f"[INITIALIZE MODEL] {init_checkpoint_path}")
        current_iter = 0

    return model, current_iter

def load_model(version:int=None) -> BigDeep:
    model = BigDeep(ActionManager()).to(Config.DEVICE)
    if version is not None:
        model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_iter_{version:03d}.pth")
    else:
        model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth")
    print(f"[LOAD MODEL] {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    return model
