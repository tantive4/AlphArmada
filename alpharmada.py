import random
import os
import shutil
from datetime import datetime
import copy
from tqdm import trange, tqdm
import gc

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import vessl

from disk_manager import DiskReplayBuffer, ArmadaDiskDataset
from configs import Config
from armada import Armada
from setup_game import setup_game
from cache_function import delete_cache
from big_deep import BigDeep
from game_encoder import encode_game_state, get_terminal_value
from para_mcts import MCTS
from action_manager import ActionManager
from action_phase import Phase
from dice import roll_dice

class AlphArmadaWorker:
    def __init__(self, model : BigDeep, worker_id : int) :
        self.worker_id = worker_id
        self.model = model
        self.model.eval()
        self.model.compile_fast_policy()

        replay_buffer_dir = Config.REPLAY_BUFFER_DIR
        if os.path.exists(replay_buffer_dir):
            shutil.rmtree(replay_buffer_dir)
        os.makedirs(os.path.dirname("output"), exist_ok=True)


        self.replay_buffer = DiskReplayBuffer(
            replay_buffer_dir, 
            Config.REPLAY_BUFFER_SIZE, 
            model.max_action_space
        )

    def self_play(self) -> None:
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
        
        # if self.worker_id == 1 :
        #     shutil.rmtree("game_visuals", ignore_errors=True)
        #     para_games[0].debuging_visual = True


        action_counter : int = 0
        saved_states : int = 0
        # with tqdm(total=Config.PARALLEL_PLAY, desc=f"[SELF-PLAY]", unit="game") as pbar:

        while any(game.winner == 0.0 for game in para_games) and action_counter < Config.MAX_GAME_STEP:
            para_indices: list[int] = [i for i, game in enumerate(para_games) if game.decision_player and game.winner == 0.0]

            if para_indices :
                # --- Perform MCTS search in parallel for decision nodes ---
                deep_search = random.random() < Config.DEEP_SEARCH_RATIO
                para_action_probs : dict[int, np.ndarray] = mcts.para_search(para_indices, deep_search=deep_search)
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
                    action = mcts.get_random_best_action(para_index, game.round)

                # Chance Node
                elif game.phase == Phase.ATTACK_ROLL_DICE:
                    if game.attack_info is None:
                        raise ValueError("No attack info for the current game phase.")
                    dice_roll = roll_dice(game.attack_info.dice_to_roll)
                    action = ('roll_dice_action', dice_roll)

                game.apply_action(action)
                mcts.advance_tree(para_index, action, game.get_snapshot())
                
                # --- Check for terminal states ---
                if game.winner != 0.0:
                    # pbar.update(1)
                    # pbar.set_postfix(last_winner=game.winner)
                    
                    saved_states += self.save_game_data(game, memory[para_index],action_counter)
                    memory[para_index].clear()
            if action_counter % 20 == 0:
                vessl.log(payload={"action_count": action_counter, "saved_states": saved_states, "ended_games" : Config.PARALLEL_PLAY - sum(1 for g in para_games if g.winner == 0.0)})
            action_counter += 1

        for game in [game for game in para_games if game.winner == 0.0]:
            with open(f'output/simulation_log.txt', 'a') as f: f.write(f"\nRuntime Warning: Game {game.para_index}\n{game.get_snapshot()}\n")

        print(f"[SELF-PLAY] saved {saved_states} states.")
        self.replay_buffer.trim_buffer()

        delete_cache()
        del para_games
        del mcts
        del initial_games
        
        # Force collection
        gc.collect()
        return

    def save_game_data(self, game : Armada, game_memory, action_count) -> int:
        """Helper to collate and save a single game's data to disk."""

        winner, aux_target = get_terminal_value(game)
        phase_list, state_list, policy_list, winner_list, aux_list = [], [], [], [], []
        end_snapshop = game.get_snapshot()

        # 'Rewind' the game using snapshots to generate encoded states
        for phase, snapshot, action_probs in game_memory:
            game.revert_snapshot(snapshot)
            encoded_state_views = encode_game_state(game)
            encoded_state_copy = {
                key: array.copy() if hasattr(array, 'copy') else array 
                for key, array in encoded_state_views.items()
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
            'ship_def_tokens': np.stack([s['ship_def_tokens'] for s in state_list]),
            'spatial': np.stack([s['spatial'] for s in state_list]),
            'relations': np.stack([s['relations'] for s in state_list]),
            'active_ship_id': np.stack([s['active_ship_id'] for s in state_list]),
            'target_ship_id': np.stack([s['target_ship_id'] for s in state_list]),
            'target_policies': np.stack(policy_list),
            'target_values': np.array(winner_list, dtype=np.float32).reshape(-1, 1),
            'target_ship_hulls': np.stack([t['ship_hulls'] for t in aux_list]),
            'target_game_length': np.stack([t['game_length'] for t in aux_list]),
            'target_win_probs': np.stack([t['win_prob'] for t in aux_list])
        }

        self.replay_buffer.add_batch(collated_data)

        with open(f'output/replay_stats.txt', 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {action_count}, {deep_search_count}, {round(winner,1)}\n")
        return deep_search_count
    
class AlphArmadaTrainer:
    def __init__(self, model : BigDeep, optimizer : optim.AdamW, num_worker: int) -> None:
        self.model = model
        self.max_action_space = model.max_action_space
        self.optimizer = optimizer
        self.num_worker = num_worker

    def train_model(self, new_checkpoint : int) -> None:

        # --- PREPARE TRAINING ---
        self.model.train()

        # --- LOAD DATASET ---
        dataset = ArmadaDiskDataset(data_root=Config.REPLAY_BUFFER_DIR)

        dataloader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
        )


        # --- TRAINING LOOP ---
        iterator = iter(dataloader)
        # for step in trange(Config.EPOCHS, desc="[TRAINING]", unit="epoch"):
        for step in range(Config.EPOCHS):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            loss = self.train(batch)
            vessl.log(step=step, payload={"training_loss": loss})

        print(f"[TRAINING] {new_checkpoint} completed. Final loss: {loss:.4f}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f'loss.txt', 'a') as f:
            f.write(f"{timestamp}, {loss:.4f}\n")


        # --- SAVE MODEL ---
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_iter_{new_checkpoint}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"[SAVE MODEL] {checkpoint_path}")

    def train(self, training_batch: dict[str, torch.Tensor]) -> float:
        """
        Trains the neural network on a batch of experiences from self-play.
        Handles split batching for MLP (Standard) and Pointer (Attention) phases.
        """
        if not training_batch:
            return 0.0
        
        # 1. Prepare Inputs
        scalar_batch : torch.Tensor = training_batch['scalar'].to(Config.DEVICE)
        ship_entity_batch : torch.Tensor = training_batch['ship_entities'].to(Config.DEVICE)
        ship_coord_batch : torch.Tensor = training_batch['ship_coords'].to(Config.DEVICE)
        ship_def_token_batch : torch.Tensor = training_batch['ship_def_tokens'].to(Config.DEVICE)
        spatial_batch : torch.Tensor = training_batch['spatial'].to(Config.DEVICE)
        relation_batch : torch.Tensor = training_batch['relations'].to(device=Config.DEVICE, dtype=torch.float32)
        active_ship_id_batch : torch.Tensor = training_batch['active_ship_id'].to(Config.DEVICE).long()
        target_ship_id_batch : torch.Tensor = training_batch['target_ship_id'].to(Config.DEVICE).long()

        phases_int_tensor = training_batch['phases']
        phases_tensor = phases_int_tensor.to(device=Config.DEVICE, dtype=torch.long)

        training_batch['target_values'] = training_batch['target_values'].to(Config.DEVICE)
        training_batch['target_policies'] = training_batch['target_policies'].to(Config.DEVICE)
        training_batch['target_ship_hulls'] = training_batch['target_ship_hulls'].to(Config.DEVICE)
        training_batch['target_game_length'] = training_batch['target_game_length'].to(Config.DEVICE)
        training_batch['target_win_probs'] = training_batch['target_win_probs'].to(Config.DEVICE)
        
        # 2. Forward Pass (Unified)
        outputs = self.model(
            scalar_input=scalar_batch,
            ship_entity_input=ship_entity_batch,
            ship_coord_input=ship_coord_batch,
            ship_token_input=ship_def_token_batch,
            spatial_input=spatial_batch,
            relation_input=relation_batch,
            active_ship_indices=active_ship_id_batch,
            target_ship_indices=target_ship_id_batch,
            phases=phases_tensor
        )
        
        # 3. Calculate Losses

        # A. Value Loss
        value_loss = F.mse_loss(outputs["value"], training_batch['target_values'], reduction='mean')

        # B. Aux Heads Loss
        hull_loss = F.mse_loss(outputs["predicted_hull"], training_batch['target_ship_hulls'], reduction='mean')
        game_len_loss = F.cross_entropy(outputs["predicted_game_length"], training_batch['target_game_length'], reduction='mean')
        win_prob_loss = F.binary_cross_entropy_with_logits(outputs["predicted_win_prob"], training_batch['target_win_probs'], reduction='mean')
        
        # C. Policy Loss
        logits = outputs["policy_logits"]
        targets = training_batch['target_policies']

        # Use F.cross_entropy which accepts soft probabilities (targets) and handles numerical stability
        policy_loss = F.cross_entropy(logits, targets, reduction='mean')

        # 4. Backpropagation
        total_loss = (
            policy_loss + 
            value_loss + 
            win_prob_loss +
            Config.HULL_LOSS_WEIGHT * hull_loss +
            Config.GAME_LENGTH_LOSS_WEIGHT * game_len_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
    