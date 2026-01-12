import random
import os
import argparse
import sys
import time
from datetime import datetime
import copy
from tqdm import trange, tqdm
import gc

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

from disk_manager import DiskReplayBuffer, ArmadaDiskDataset
from configs import Config
from armada import Armada
from setup_game import setup_game
from cache_function import delete_cache
from jit_geometry import pre_compile_jit_geometry
from armada_net import ArmadaNet
from game_encoder import encode_game_state, get_terminal_value
from para_mcts import MCTS
from action_manager import ActionManager
from action_phase import Phase, get_action_str
from dice import roll_dice


class AlphArmada:
    def __init__(self, model : ArmadaNet, optimizer : optim.AdamW) :
        self.model : ArmadaNet = model
        self.optimizer : optim.AdamW = optimizer

        self.max_action_space = model.max_action_space
        self.replay_buffer = DiskReplayBuffer(Config.REPLAY_BUFFER_DIR, Config.REPLAY_BUFFER_SIZE, self.max_action_space)
        
    def para_self_play(self) -> None:
        memory : dict[int, list[tuple[Phase, tuple, np.ndarray]]] = {para_index : list() for para_index in range(Config.PARALLEL_PLAY)}

        action_manager = ActionManager()
        initial_games = [setup_game() for _ in range(Config.PARALLEL_DIVERSE_FACTOR)]
        para_games: list[Armada] = [
            copy.deepcopy(game) 
            for game in initial_games 
            for _ in range(Config.PARALLEL_SAME_GAME)
        ]
        for para_index, para_game in enumerate(para_games):
            para_game.para_index = para_index
        mcts : MCTS = MCTS(copy.deepcopy(para_games), action_manager, self.model)
        
        if os.path.exists("game_visuals"):import shutil; shutil.rmtree("game_visuals")
        para_games[0].debuging_visual = True


        action_counter : int = 0
        with tqdm(total=Config.PARALLEL_PLAY, desc=f"[SELF-PLAY]", unit="game") as pbar:
            while any(game.winner == 0.0 for game in para_games) and action_counter < Config.MAX_GAME_STEP:
                simulation_players: dict[int, int] = {i: game.decision_player for i, game in enumerate(para_games) if game.decision_player and game.winner == 0.0}

                if simulation_players :
                    # --- Perform MCTS search in parallel for decision nodes ---
                    deep_search = random.random() < Config.DEEP_SEARCH_RATIO
                    para_action_probs : dict[int, np.ndarray] = mcts.para_search(simulation_players, deep_search=deep_search)
                    if deep_search :
                        for para_index in para_action_probs:
                            game : Armada = para_games[para_index]
                            snapshot = game.get_snapshot()
                            memory[para_index].append((game.phase, snapshot, para_action_probs[para_index]))
                
                # --- Process all games (decision and non-decision) for one step ---
                for para_index in range(Config.PARALLEL_PLAY) :
                    game : Armada = para_games[para_index]
                    if game.winner != 0.0:
                        continue

                    if game.decision_player :
                        action = mcts.get_random_best_action(para_index, game.decision_player, game.round)

                    # Chance Node
                    elif game.phase == Phase.ATTACK_ROLL_DICE:
                        if game.attack_info is None:
                            raise ValueError("No attack info for the current game phase.")
                        dice_roll = roll_dice(game.attack_info.dice_to_roll)
                        action = ('roll_dice_action', dice_roll)

                    # Information Set Node
                    elif game.phase == Phase.SHIP_REVEAL_COMMAND_DIAL:
                        if len(game.get_valid_actions()) != 1:
                            raise ValueError("Multiple valid actions in information set node.")
                        action = game.get_valid_actions()[0]
                    game.apply_action(action)
                    mcts.advance_tree(para_index, action, game.get_snapshot())
                    
                    # --- Check for terminal states ---
                    if game.winner != 0.0:
                        pbar.update(1)
                        pbar.set_postfix(last_winner=game.winner)
                        self.save_game_data(game, memory[para_index],action_counter)
                        memory[para_index].clear()

                action_counter += 1

        for game in [game for game in para_games if game.winner == 0.0]:
            with open('simulation_log.txt', 'a') as f: f.write(f"\nRuntime Warning: Game {game.para_index}\n{game.get_snapshot()}\n")
        
        delete_cache()
        del para_games
        del mcts
        del initial_games
        
        # Force collection
        gc.collect()
        return

    def save_game_data(self, game : Armada, game_memory, action_count) -> None:
        """Helper to collate and save a single game's data to disk."""

        winner, aux_target = get_terminal_value(game)
        phase_list, state_list, policy_list, winner_list, aux_list = [], [], [], [], []
        end_snapshop = game.get_snapshot()

        # 'Rewind' the game using snapshots to generate encoded states
        for phase, snapshot, action_probs in game_memory:
            game.revert_snapshot(snapshot)
            encoded_state_views = encode_game_state(game)
            encoded_state_copy = {
                key: array.copy() for key, array in encoded_state_views.items()
            }
            
            phase_list.append(phase)
            state_list.append(encoded_state_copy)
            policy_list.append(action_probs)
            winner_list.append(winner)
            aux_list.append(aux_target)
        
        game.revert_snapshot(end_snapshop)
        deep_search_count = len(phase_list)

        collated_data = {
            'phases': list(phase_list),
            'scalar': np.stack([s['scalar'] for s in state_list]),
            'ship_entities': np.stack([s['ship_entities'] for s in state_list]),
            'ship_coords': np.stack([s['ship_coords'] for s in state_list]),
            'spatial': np.stack([s['spatial'] for s in state_list]),
            'relations': np.stack([s['relations'] for s in state_list]),
            'target_policies': np.stack(policy_list),
            'target_values': np.array(winner_list, dtype=np.float32).reshape(-1, 1),
            'target_ship_hulls': np.stack([t['ship_hulls'] for t in aux_list]),
            'target_game_length': np.stack([t['game_length'] for t in aux_list])
        }

        self.replay_buffer.add_batch(collated_data)

        with open('replay_stats.txt', 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {action_count}, {deep_search_count}, {round(winner,1)}\n")

    def train(self, training_batch: dict[str, torch.Tensor]) -> float:
        """
        Trains the neural network on a batch of experiences from self-play.
        Handles split batching for MLP (Standard) and Pointer (Attention) phases.
        """
        if not training_batch:
            return 0.0
        
        # 1. Prepare Inputs
        target_values_tensor : torch.Tensor = training_batch['target_values'].to(Config.DEVICE)
        scalar_batch : torch.Tensor = training_batch['scalar'].to(Config.DEVICE)
        ship_entity_batch : torch.Tensor = training_batch['ship_entities'].to(Config.DEVICE)
        ship_coord_batch : torch.Tensor = training_batch['ship_coords'].to(Config.DEVICE)
        spatial_batch : torch.Tensor = training_batch['spatial'].to(Config.DEVICE)
        relation_batch : torch.Tensor = training_batch['relations'].to(device=Config.DEVICE, dtype=torch.float32)

        phases_int_tensor = training_batch['phases']
        phases_tensor = phases_int_tensor.to(device=Config.DEVICE, dtype=torch.long)
        phases = [Phase(p.item()) for p in phases_int_tensor]

        training_batch['target_ship_hulls'] = training_batch['target_ship_hulls'].to(Config.DEVICE)
        training_batch['target_game_length'] = training_batch['target_game_length'].to(Config.DEVICE)
        training_batch['target_policies'] = training_batch['target_policies'].to(Config.DEVICE)


        batch_size = len(phases)

        # 2. Identify Indices for MLP vs Pointer Phases
        pointer_phases = {Phase.SHIP_ACTIVATE.value, Phase.SHIP_CHOOSE_TARGET_SHIP.value}
        
        # Create boolean mask on CPU/GPU
        is_pointer_tensor = torch.zeros(batch_size, dtype=torch.bool, device=Config.DEVICE)
        for val in pointer_phases:
            is_pointer_tensor |= (phases_tensor == val)
            
        ptr_indices = torch.nonzero(is_pointer_tensor, as_tuple=True)[0]
        mlp_indices = torch.nonzero(~is_pointer_tensor, as_tuple=True)[0]

        # Containers for outputs to aggregate loss later
        results = {
            'value_loss': torch.tensor(0.0, device=Config.DEVICE),
            'hull_loss': torch.tensor(0.0, device=Config.DEVICE),
            'game_length_loss': torch.tensor(0.0, device=Config.DEVICE),
            'policy_loss': torch.tensor(0.0, device=Config.DEVICE),
        }

        # Helper to accumulate MSE/CrossEntropy losses from partial batches
        def accumulate_standard_losses(model_out, indices):
            # Value
            pred_val = model_out["value"]
            target_val = target_values_tensor[indices]
            results['value_loss'] += F.mse_loss(pred_val, target_val, reduction='sum')

            # Aux Heads
            results['hull_loss'] += F.mse_loss(model_out["predicted_hull"], training_batch['target_ship_hulls'][indices], reduction='sum')

            # Game Length
            results['game_length_loss'] += F.cross_entropy(model_out["predicted_game_length"], training_batch['target_game_length'][indices], reduction='sum')

        # --- 3. Forward Pass: Standard (MLP) Phases ---
        if len(mlp_indices) > 0:
            # Slice inputs
            mlp_out = self.model(
                scalar_batch[mlp_indices],
                ship_entity_batch[mlp_indices],
                ship_coord_batch[mlp_indices],
                spatial_batch[mlp_indices],
                relation_batch[mlp_indices],
                phases_tensor[mlp_indices]
            )
            
            accumulate_standard_losses(mlp_out, mlp_indices)
            
            # Policy Loss (Per Phase Grouping)
            logits = mlp_out["policy_logits"]
            subset_phases = [phases[i] for i in mlp_indices.cpu().numpy()]
            subset_targets = training_batch['target_policies'][mlp_indices]
            
            # Group by Phase Name to handle masking
            phase_map = {}
            for idx, p in enumerate(subset_phases):
                if p.name not in phase_map: phase_map[p.name] = []
                phase_map[p.name].append(idx)
            
            for p_name, p_idxs in phase_map.items():
                # Get logits for this specific phase group
                group_logits = logits[p_idxs]
                group_targets = subset_targets[p_idxs]
                
                # Slice to valid action size
                action_size = len(self.model.action_manager.get_action_map(Phase[p_name]))
                group_logits = group_logits[:, :action_size]
                group_targets = group_targets[:, :action_size]
                
                log_probs = F.log_softmax(group_logits, dim=1)
                results['policy_loss'] += -(group_targets * log_probs).sum(dim=1).sum()

        # --- 4. Forward Pass: Pointer Phases ---
        if len(ptr_indices) > 0:
            ptr_out = self.model(
                scalar_batch[ptr_indices],
                ship_entity_batch[ptr_indices],
                ship_coord_batch[ptr_indices], # Corrected arg
                spatial_batch[ptr_indices],
                relation_batch[ptr_indices],
                phases_tensor[ptr_indices]
            )
            
            accumulate_standard_losses(ptr_out, ptr_indices)
            
            # Policy Loss (Pointer)
            logits = ptr_out["policy_logits"] # Shape [B, N+1]
            targets = training_batch['target_policies'][ptr_indices] # Shape [B, Max_Action]
            
            # Slice targets to match Pointer Head size (N+1)
            # Assuming MCTS targets for these phases are stored in indices 0..N
            valid_size = logits.shape[1] 
            targets = targets[:, :valid_size]
            
            log_probs = F.log_softmax(logits, dim=1)
            results['policy_loss'] += -(targets * log_probs).sum(dim=1).sum()


        # --- 5. Finalize Loss ---
        # Average over total batch size
        total_loss = (
            results['policy_loss'] / batch_size + 
            results['value_loss'] / batch_size + 
            Config.HULL_LOSS_WEIGHT * (results['hull_loss'] / batch_size) +
            Config.GAME_LENGTH_LOSS_WEIGHT * (results['game_length_loss'] / batch_size)
        )

        # L2 Regularization
        l2_reg = torch.tensor(0., device=Config.DEVICE)
        for param in self.model.parameters():
            l2_reg += torch.sum(param.pow(2))
        
        total_loss += Config.L2_LAMBDA * l2_reg

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def learn(self, current_iteration):
            """
            Main training loop:
            1. Perform self-play to generate training data & save to disk.
            2. Load data from disk into DataLoader.
            3. Train the model on the data.
            4. Save the updated model checkpoint.
            """

            # --- SELF PLAY & SAVE DATA ---
            self.model.eval()
            self.model.compile_fast_policy()
            self.para_self_play()

            

            # --- PREPARE TRAINING ---
            self.model.train()
            self.model.fast_policy_ready = False
            if hasattr(self.model, 'w1_stack'):
                del self.model.w1_stack, self.model.b1_stack, self.model.w2_stack, self.model.b2_stack, self.model.w3_stack, self.model.b3_stack

            dataset = ArmadaDiskDataset(
                data_dir=Config.REPLAY_BUFFER_DIR, 
                max_size=Config.REPLAY_BUFFER_SIZE, 
                current_size=self.replay_buffer.current_size, 
                action_space_size=self.max_action_space
            )
            if self.replay_buffer.current_size < Config.REPLAY_BUFFER_SIZE // 2:
                print(f"[TRAINING] Not enough data to train. Current size: {self.replay_buffer.current_size} / {Config.REPLAY_BUFFER_SIZE}")
                return

            dataloader = DataLoader(
                dataset, 
                batch_size=Config.BATCH_SIZE, 
                shuffle=True, 
                num_workers=4, 
                # pin_memory=True
            )


            # --- TRAINING LOOP ---
            iterator = iter(dataloader)
            self.optimizer.zero_grad()
            for step in trange(Config.EPOCHS, desc="[TRAINING]", unit="epoch"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    batch = next(iterator)
                loss = self.train(batch)

            print(f"[TRAINING] {current_iteration+1} completed. Final loss: {loss:.4f}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('loss.txt', 'a') as f:
                f.write(f"{timestamp}, {loss:.4f}\n")


            # --- SAVE MODEL ---
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_iter_{current_iteration + 1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"[SAVE MODEL] {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True, help="Current iteration number")
    args = parser.parse_args()
    
    current_iter = args.iter

    # Initialize the model and optimizer
    model = ArmadaNet(ActionManager()).to(Config.DEVICE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Find all checkpoint files
    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('model_iter_') and f.endswith('.pth')]
    
    if checkpoints:
        # Find the checkpoint with the highest iteration number
        latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint_file)
        print(f"[LOAD MODEL] {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))

    else:
        init_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_iter_0.pth")
        torch.save(model.state_dict(), init_checkpoint_path)
        print(f"[INITAIIZE MODEL] {init_checkpoint_path}")
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Create the training manager and start the learning process
    alpharmada_trainer = AlphArmada(model, optimizer)
    alpharmada_trainer.learn(current_iteration=current_iter)

if __name__ == "__main__":
    main()