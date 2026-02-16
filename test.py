import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import shutil
import numpy as np

from configs import Config
from big_deep import BigDeep
from action_manager import ActionManager
from disk_manager import ArmadaChunkDataset

# --- FORCE CPU FOR DEBUGGING ---
Config.DEVICE = 'cpu'
torch.manual_seed(42)
np.random.seed(42)

def check_outputs(outputs, label):
    """Checks model outputs for NaNs."""
    for k, v in outputs.items():
        if torch.isnan(v).any():
            print(f"❌ {label}: NaN detected in output '{k}'")
            return False
        if torch.isinf(v).any():
            print(f"❌ {label}: Inf detected in output '{k}'")
            return False
    print(f"✅ {label}: Outputs Clean")
    return True

def get_weight_stats(model):
    """Returns max/mean of all weights to track explosion."""
    max_val = 0.0
    mean_val = 0.0
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            max_val = max(max_val, param.abs().max().item())
            mean_val += param.abs().mean().item()
            count += 1
    return max_val, (mean_val / count) if count > 0 else 0

def run_verification():
    print(f"{'='*60}")
    print("VERIFICATION: Single Batch Overfit Test")
    print(f"{'='*60}")

    # 1. Load Clean Model (Iter 0)
    print("\n[1] Loading model_iter_000.pth...")
    am = ActionManager()
    model = BigDeep(am).to(Config.DEVICE)
    
    init_path = os.path.join(Config.CHECKPOINT_DIR, "model_iter_000.pth")
    if not os.path.exists(init_path):
        print(f"❌ Error: {init_path} not found. Please ensure iter_0 exists.")
        return
    model.load_state_dict(torch.load(init_path, map_location=Config.DEVICE))
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.L2_LAMBDA)

    # 2. Prepare Fixed Batch (128 samples)
    print("[2] Loading fixed batch of 128 samples...")
    dataset = ArmadaChunkDataset(data_root=Config.REPLAY_BUFFER_DIR, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0) # 1 chunk = 128 samples
    
    try:
        raw_batch = next(iter(dataloader))
    except StopIteration:
        print("❌ Error: Replay buffer empty.")
        return

    # Flatten [1, 128, ...] -> [128, ...] and move to device
    batch = {k: v.flatten(0, 1).to(Config.DEVICE) for k, v in raw_batch.items()}
    print(f"    Batch Size: {batch['scalar'].shape[0]}")

    # 3. Step 1: Pre-Update Check
    print("\n[3] PRE-UPDATE Forward Pass (Eval Mode)...")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                scalar_input=batch['scalar'],
                ship_entity_input=batch['ship_entities'],
                ship_coord_input=batch['ship_coords'],
                ship_token_input=batch['ship_def_tokens'],
                spatial_input=batch['spatial'],
                relation_input=batch['relations'],
                active_ship_indices=batch['active_ship_id'],
                target_ship_indices=batch['target_ship_id'],
                phases=batch['phases']
            )
            if not check_outputs(outputs, "Pre-Update"): return
        except ValueError as e:
            print(f"❌ Pre-Update Crash: {e}")
            return
    

    print("[4] PRE-UPDATE Forward Pass (Train Mode)")
    model.train()
    
    # Forward
    outputs = model(
        scalar_input=batch['scalar'],
        ship_entity_input=batch['ship_entities'],
        ship_coord_input=batch['ship_coords'],
        ship_token_input=batch['ship_def_tokens'],
        spatial_input=batch['spatial'],
        relation_input=batch['relations'],
        active_ship_indices=batch['active_ship_id'],
        target_ship_indices=batch['target_ship_id'],
        phases=batch['phases']
    )
    
    # Loss calc
    value_loss = F.mse_loss(outputs["value"], batch['target_values'])
    hull_loss = F.mse_loss(outputs["predicted_hull"], batch['target_ship_hulls'])
    game_len_loss = F.cross_entropy(outputs["predicted_game_length"], batch['target_game_length'])
    win_prob_loss = F.binary_cross_entropy_with_logits(outputs["predicted_win_prob"], batch['target_win_probs'])
    policy_loss = F.cross_entropy(outputs["policy_logits"], batch['target_policies'])

    total_loss = (
        policy_loss + value_loss + win_prob_loss +
        Config.HULL_LOSS_WEIGHT * hull_loss +
        Config.GAME_LENGTH_LOSS_WEIGHT * game_len_loss
    )
    
    print(f"    Loss: {total_loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    
    # Clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Stats before step
    w_max_old, w_mean_old = get_weight_stats(model)
    
    # Step
    optimizer.step()
    
    # Stats after step
    w_max_new, w_mean_new = get_weight_stats(model)
    print(f"    Weight Stats: Max {w_max_old:.4f} -> {w_max_new:.4f} | Mean {w_mean_old:.4f} -> {w_mean_new:.4f}")
    # 6. Step 4: Save and Reload
    print("\n[5] Saving and Reloading 'model_verify_temp.pth'...")
    temp_path = os.path.join(Config.CHECKPOINT_DIR, "model_verify_temp.pth")
    torch.save(model.state_dict(), temp_path)

    print("[7] RELOADED Forward Pass...(Train Mode)")
    model_new1 = BigDeep(am).to(Config.DEVICE)
    model_new1.load_state_dict(torch.load(temp_path, map_location=Config.DEVICE))
    model_new1.train()

    outputs = model_new1(
    scalar_input=batch['scalar'],
    ship_entity_input=batch['ship_entities'],
    ship_coord_input=batch['ship_coords'],
    ship_token_input=batch['ship_def_tokens'],
    spatial_input=batch['spatial'],
    relation_input=batch['relations'],
    active_ship_indices=batch['active_ship_id'],
    target_ship_indices=batch['target_ship_id'],
    phases=batch['phases']
    )

    print('train mode pass')
    
    # Load into NEW model instance
    model_new = BigDeep(am).to(Config.DEVICE)
    model_new.load_state_dict(torch.load(temp_path, map_location=Config.DEVICE))
    
    print("\n[8] RELOADED Forward Pass...")
    model_new.eval()
    with torch.no_grad():
        outputs = model_new(
            scalar_input=batch['scalar'],
            ship_entity_input=batch['ship_entities'],
            ship_coord_input=batch['ship_coords'],
            ship_token_input=batch['ship_def_tokens'],
            spatial_input=batch['spatial'],
            relation_input=batch['relations'],
            active_ship_indices=batch['active_ship_id'],
            target_ship_indices=batch['target_ship_id'],
            phases=batch['phases']
        )
    check_outputs(outputs, "Post-Update (Eval)")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    run_verification()