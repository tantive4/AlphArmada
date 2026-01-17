import torch
import torch.nn.functional as F
import numpy as np

import os

from configs import Config
from action_space import _make_hashable
from action_phase import Phase, get_action_str

from game_encoder import encode_game_state


def model_check(game, model, action_manager):
    """
    Evaluates the ArmadaNet model on a single game instance and returns
    both raw and masked policy outputs.
    
    Args:
        game (Armada): The game object representing the current state.
        model (ArmadaNet): The neural network model.
        action_manager (ActionManager): The manager handling action-to-index mapping.
        
    Returns:
        dict: A dictionary containing:
            - 'raw_logits': Tensor [1, ActionSpace], raw output from the model.
            - 'masked_logits': Tensor [1, ActionSpace], logits with invalid actions masked to -inf.
            - 'probs': Tensor [1, ActionSpace], softmax probabilities of masked logits.
            - 'value': float, the value head prediction (-1 to 1).
            - 'valid_actions': list, the list of valid action tuples from the game state.
    """
    model.eval()
    
    # --- 1. Encode the Game State ---
    # This updates the numpy arrays stored within the game object and returns a dictionary of views.
    encoded_dict = encode_game_state(game)
    
    # --- 2. Prepare Tensor Inputs ---
    # Convert numpy arrays to tensors and add Batch Dimension [1, ...]
    scalar_input = torch.tensor(encoded_dict['scalar'], dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
    ship_entity_input = torch.tensor(encoded_dict['ship_entities'], dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
    ship_coord_input = torch.tensor(encoded_dict['ship_coords'], dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
    spatial_input = torch.tensor(encoded_dict['spatial'], dtype=torch.uint8, device=Config.DEVICE).unsqueeze(0)
    relation_input = torch.tensor(encoded_dict['relations'], dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
    
    # Indices must be LongTensor
    active_ship_idx = torch.tensor([encoded_dict['active_ship_id']], dtype=torch.long, device=Config.DEVICE)
    phases_tensor = torch.tensor([game.phase.value], dtype=torch.long, device=Config.DEVICE)
    
    # --- 3. Model Inference ---
    with torch.no_grad():
        outputs = model(
            scalar_input, 
            ship_entity_input, 
            ship_coord_input, 
            spatial_input, 
            relation_input, 
            active_ship_idx, 
            phases_tensor
        )
        
    raw_logits = outputs['policy_logits'] # Shape: [1, OutputDim]
    value_pred = outputs['value'].item()
    
    # --- 4. Apply Masking Logic (Mirroring _mask_policy) ---
    
    # A. Calculate initial probabilities from raw logits
    policy = F.softmax(raw_logits, dim=1).squeeze(0) # [ActionSpace]
    
    # B. Initialize Action Mask (Zeros)
    action_mask_tensor = torch.zeros_like(policy, dtype=torch.float32)
    
    # C. Get Valid Actions and Map to Indices
    valid_actions = game.get_valid_actions()
    action_map = action_manager.get_action_map(game.phase)
    
    for action in valid_actions:
        # Reconstruct the hashable key: (action_name, hashable_value)
        key = (action[0], _make_hashable(action[1]))
        
        if key in action_map:
            idx = action_map[key]
            # Safety check: Ensure index is within model output bounds
            if idx < action_mask_tensor.shape[0]:
                action_mask_tensor[idx] = 1.0
        else:
            print(f"Warning: Valid game action {key} not found in ActionManager map for phase {game.phase}")

    # D. Mask and Re-normalize
    # "policy_sum += policy[action_index]" (equivalent to sum of masked policy)
    masked_policy = policy * action_mask_tensor
    policy_sum = masked_policy.sum()
    
    if policy_sum > 0:
        masked_policy /= policy_sum
    else:
        # Handle case where network assigns 0 probability to all valid moves
        print(f"Warning: Zero policy sum for valid moves. Using uniform distribution.")
        num_valid = len(valid_actions)
        if num_valid > 0:
            masked_policy = action_mask_tensor / num_valid

    # E. Create "Masked Logits" just for visualization/debugging (setting invalid to -inf)
    masked_logits = raw_logits.clone()
    masked_logits[0, action_mask_tensor == 0] = float('-inf')
    
    return {
        'raw_logits': raw_logits,
        'masked_logits': masked_logits,
        'probs': masked_policy.unsqueeze(0), # [1, ActionSpace]
        'value': value_pred,
        'valid_actions': valid_actions
    }

if __name__ == "__main__":
    from armada_net import ArmadaNet
    from action_manager import ActionManager
    from setup_game import setup_game
    import random
    random.seed(6)
    
    am = ActionManager()
    model = ArmadaNet(am).to(Config.DEVICE)
    
    # Automatic Checkpoint Loading
    if os.path.exists(Config.CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('model_iter_') and f.endswith('.pth')]
        
        if checkpoints:
            latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint_file)
            print(f"[LOAD MODEL] {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        else:
            print("[LOAD MODEL] No checkpoints found. Using random initialized weights.")
    else:
        print(f"[LOAD MODEL] Checkpoint directory '{Config.CHECKPOINT_DIR}' not found. Using random initialized weights.")

    if os.path.exists("game_visuals"):import shutil; shutil.rmtree("game_visuals")
    game = setup_game(debuging_visual=True)
    game.apply_action(('activate_ship_action', 0))
    
    result = model_check(game, model, am)
    
    # --- Display Results ---
    print("\n" + "="*40)
    print(f"Phase: {game.phase.name}")
    print(f"Value Head Prediction: {result['value']:.4f}")
    print(f"Valid Actions Available: {len(result['valid_actions'])}")
    print("-" * 40)
    
    print("Top 5 Predicted Actions:")
    probs = result['probs'][0]
    top_k = torch.topk(probs, k=min(5, probs.shape[0]))
    
    # Get mapping from index to action tuple
    action_map = am.get_action_map(game.phase)
    idx_to_action = {v: k for k, v in action_map.items()}
    
    for i in range(len(top_k.indices)):
        idx = top_k.indices[i].item()
        score = top_k.values[i].item()
        
        # Display only if probability > 0
        if score > 0:
            action_tuple = idx_to_action[idx]
            action_str = get_action_str(game, action_tuple)
            print(f"  {action_str} : {score:.4f}")