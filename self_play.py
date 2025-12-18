from collections import deque
import random
import os
import time
from datetime import datetime
import copy
from tqdm import trange, tqdm
import gc

import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np

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
        
    def para_self_play(self, iter_num: int = 0, batch_num: int = 0) -> None:
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
        with tqdm(total=Config.PARALLEL_PLAY, desc=f"Self-Play Batch {batch_num + 1}", unit="game") as pbar:
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
                    # with open(f'simulation_log{para_index+1}.txt', 'a') as f: f.write(f"Game {para_index+1} Round {game.round} Phase: {game.phase.name}, Action: {get_action_str(game, action)}\n")
                    # if para_index == 0: print(f"Game 1 Round {game.round} Phase: {game.phase.name}, Action: {get_action_str(game, action)}")
                    game.apply_action(action)
                    mcts.advance_tree(para_index, action, game.get_snapshot())
                    
                    # --- Check for terminal states ---
                    if game.winner != 0.0:
                        pbar.update(1)
                        pbar.set_postfix(last_round=f"{game.round:.1f}", last_winner=int(game.winner))
                        winner, aux_target = get_terminal_value(game)
                        game_data_buffer = []
                        end_snapshop = game.get_snapshot()
                        # 'Rewind' the game using snapshots to generate encoded states
                        for phase, snapshot, action_probs in memory[para_index]:
                            game.revert_snapshot(snapshot)
                            
                            # Now we encode, only when needed
                            encoded_state_views = encode_game_state(game)
                            encoded_state_copy = {
                                key: array.copy() for key, array in encoded_state_views.items()
                            }
                            
                            game_data_buffer.append((phase, encoded_state_copy, action_probs, winner, aux_target))
                        
                        game.revert_snapshot(end_snapshop)
                        # Clear memory for this game immediately
                        memory[para_index].clear()

                        # Save this specific game's data immediately to disk
                        if game_data_buffer:
                            self.save_game_data(game_data_buffer, iter_num, batch_num, para_index)
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

    def save_game_data(self, game_data, iter_num, batch_num, para_index):
        """Helper to collate and save a single game's data to disk."""
        phases, states, action_probs, winners, aux_targets = zip(*game_data)
        
        collated_data = {
            'phases': list(phases),
            'scalar': np.stack([s['scalar'] for s in states]),
            'ship_entities': np.stack([s['ship_entities'] for s in states]),
            'squad_entities': np.stack([s['squad_entities'] for s in states]),
            'spatial': np.stack([s['spatial'] for s in states]),
            'relations': np.stack([s['relations'] for s in states]),
            'target_policies': np.stack(action_probs),
            'target_values': np.array(winners, dtype=np.float32),
            'target_ship_hulls': np.stack([t['ship_hulls'] for t in aux_targets]),
            'target_squad_hulls': np.stack([t['squad_hulls'] for t in aux_targets]),
            'target_game_length': np.stack([t['game_length'] for t in aux_targets])
        }

        timestamp = int(time.time() * 1000)
        filename = f"replay_{timestamp}_iter_{iter_num}_batch_{batch_num}_game_{para_index}.pth"
        path = os.path.join(Config.REPLAY_BUFFER_DIR, filename)
        torch.save(collated_data, path)


    def train(self, training_batch: dict[str, torch.Tensor]) -> float:
        """
        Trains the neural network on a batch of experiences from self-play.
        """
        if not training_batch:
            return 0.0

        
        
        # Prepare batched tensors
        target_values_tensor : torch.Tensor = training_batch['target_values']
        scalar_batch : torch.Tensor = training_batch['scalar'] 
        ship_entity_batch : torch.Tensor = training_batch['ship_entities'] 
        squad_entity_batch : torch.Tensor = training_batch['squad_entities'] 
        spatial_batch : torch.Tensor = training_batch['spatial'] 
        relation_batch : torch.Tensor = training_batch['relations'] 
        phases : list[Phase] = training_batch['phases'] #type: ignore
        
        # --- Single Batched Forward Pass ---
        model_output = self.model(
            scalar_batch,
            ship_entity_batch,
            squad_entity_batch,
            spatial_batch,
            relation_batch,
            list(phases)
        )
        policy_logits = model_output["policy_logits"]
        value_pred = model_output["value"]


        # --- Calculate Losses ---
        
        # Value loss can be calculated on the whole batch at once
        value_loss = F.mse_loss(value_pred, target_values_tensor)

        # Policy loss needs to be calculated per phase group due to different shapes
        policy_loss = torch.tensor(0.0, device=Config.DEVICE)
        
        # Group indices by phase
        phase_indices :dict[str, list[int]] = {}
        for para_index, phase in enumerate(phases):
            if phase.name not in phase_indices:
                phase_indices[phase.name] = []
            phase_indices[phase.name].append(para_index)

        # Get the pre-batched tensor
        target_policies_tensor = training_batch['target_policies']

        for phase_name, indices in phase_indices.items():
            if not indices:
                continue
            
            group_logits = policy_logits[indices]
            # Select from the main tensor instead of stacking
            group_target_policies = target_policies_tensor[indices]
            
            action_space_size = group_target_policies.shape[1]
            group_logits_sliced = group_logits[:, :action_space_size]
            
            log_probs = F.log_softmax(group_logits_sliced, dim=1)
            policy_loss += -(group_target_policies * log_probs).sum(dim=1).mean()

        # --- Auxiliary Losses ---
        target_hull : torch.Tensor = training_batch['target_ship_hulls']
        hull_loss = F.mse_loss(model_output["predicted_hull"], target_hull)

        target_squads : torch.Tensor = training_batch['target_squad_hulls']
        squad_loss = F.mse_loss(model_output["predicted_squads"], target_squads)

        target_game_length : torch.Tensor = training_batch['target_game_length']
        game_length_loss = F.cross_entropy(model_output["predicted_game_length"], target_game_length)

        # L2 Regularization
        l2_reg = torch.tensor(0., device=Config.DEVICE)
        for param in self.model.parameters():
            l2_reg += torch.sum(param.pow(2))

        # Combine losses
        total_loss = (
            policy_loss + 
            value_loss + 
            Config.HULL_LOSS_WEIGHT * hull_loss +
            Config.SQUAD_LOSS_WEIGHT * squad_loss +
            Config.GAME_LENGTH_LOSS_WEIGHT * game_length_loss +
            Config.L2_LAMBDA * l2_reg
        )

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
    

    def learn(self, start_iteration=0):
            """
            Main training loop:
            1. Generates self-play data in batches for the current iteration, saving each batch to a file.
            2. Before training, it loads data from all past iterations, starting with the most recent,
            until the replay buffer is full.
            3. Trains the model on this buffer of recent experiences.
            """
            
            # Create directories if they don't exist
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(Config.REPLAY_BUFFER_DIR, exist_ok=True)

            for i in range(start_iteration, Config.ITERATIONS):
                print(f"\n----- Iteration {i+1}/{Config.ITERATIONS} -----\n")
                
                # --- Step 1 & 2: Start fresh self-play and save batches ---
                self.model.eval()
                self.model.compile_fast_policy()
                print("Starting self-play phase...")
                for self_play_iteration in range(Config.SELF_PLAY_GAMES):

                    self.para_self_play(iter_num=i, batch_num=self_play_iteration)
                    
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    print(f"Self-play batch {self_play_iteration + 1}/{Config.SELF_PLAY_GAMES} completed.")

                
                # --- Step 3: Before training, load all previous self-play data ---
                print("\n--- Preparing for training phase: Loading replay buffers... ---")

                # This dict will hold all data in concatenated form
                batch_array_dict = {
                    'phases': [], 'scalar': [], 'ship_entities': [], 'squad_entities': [],
                    'spatial': [], 'relations': [], 'target_policies': [], 'target_values': [],
                    'target_ship_hulls': [], 'target_squad_hulls': [], 'target_game_length': []
                }
                total_samples = 0

                all_replay_files = [file for file in os.listdir(Config.REPLAY_BUFFER_DIR) if file.endswith('.pth')]
                all_replay_files.sort(reverse=True) # Load newest first

                for filename in all_replay_files:
                    if total_samples >= Config.REPLAY_BUFFER_SIZE:
                        break # Stop if buffer is full
                    
                    replay_buffer_path = os.path.join(Config.REPLAY_BUFFER_DIR, filename)
                    try:
                        # Load one collated batch dictionary
                        batch_dict = torch.load(replay_buffer_path, weights_only=False)

                        # Append the data slices
                        batch_array_dict['phases'].extend(batch_dict['phases'])
                        # Iterate all keys *except* 'phases'
                        for key in list(batch_array_dict.keys())[1:]: 
                            batch_array_dict[key].append(batch_dict[key])
                        
                        total_samples += len(batch_dict['phases'])

                    except (EOFError, pickle.UnpicklingError):
                        print(f"Warning: Could not load {replay_buffer_path}. File might be corrupted. Skipping.")

                if total_samples < Config.REPLAY_BUFFER_SIZE:
                    print("Not enough samples to form a batch. Skipping training for this iteration.")
                    continue

                # --- Concatenate all chunks into single giant arrays ---
                print(f"Loaded {total_samples} total experiences.")
                total_array_dict = {}
                total_array_dict['phases'] = batch_array_dict['phases'] # Already a flat list
                for key in list(batch_array_dict.keys())[1:]:
                    total_array_dict[key] = np.concatenate(batch_array_dict[key], axis=0)

                # --- Step 4: Start training on the loaded data ---
                print("Starting training phase...")
                self.model.train()
                self.model.fast_policy_ready = False
                if hasattr(self.model, 'w1_stack'):
                    del self.model.w1_stack
                    del self.model.b1_stack
                    del self.model.w2_stack
                    del self.model.b2_stack
                    del self.model.w3_stack
                    del self.model.b3_stack

                    
                self.optimizer.zero_grad()
                for step in trange(Config.EPOCHS):
                    # 1. Sample BATCH_SIZE random indices
                    indices = torch.randint(0, total_samples, (Config.BATCH_SIZE,))
                    
                    # 2. Create the batch *instantly* via tensor indexing
                    training_batch_tensors = {
                        'phases': [total_array_dict['phases'][i] for i in indices],
                        'scalar': torch.from_numpy(total_array_dict['scalar'][indices]).float().to(Config.DEVICE),
                        'ship_entities': torch.from_numpy(total_array_dict['ship_entities'][indices]).float().to(Config.DEVICE),
                        'squad_entities': torch.from_numpy(total_array_dict['squad_entities'][indices]).float().to(Config.DEVICE),
                        'spatial': torch.from_numpy(total_array_dict['spatial'][indices]).float().to(Config.DEVICE),
                        'relations': torch.from_numpy(total_array_dict['relations'][indices]).float().to(Config.DEVICE),
                        'target_policies': torch.from_numpy(total_array_dict['target_policies'][indices]).float().to(Config.DEVICE),
                        'target_values': torch.from_numpy(total_array_dict['target_values'][indices]).float().to(Config.DEVICE).view(-1, 1),
                        'target_ship_hulls': torch.from_numpy(total_array_dict['target_ship_hulls'][indices]).float().to(Config.DEVICE),
                        'target_squad_hulls': torch.from_numpy(total_array_dict['target_squad_hulls'][indices]).float().to(Config.DEVICE),
                        'target_game_length': torch.from_numpy(total_array_dict['target_game_length'][indices]).float().to(Config.DEVICE),
                    }
                    
                    # 3. Pass this ready-to-use batch to train
                    loss = self.train(training_batch_tensors)
                    
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print(f"Iteration {i+1} completed. Final training loss: {loss:.4f}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('loss.txt', 'a') as f:
                    f.write(f"{timestamp}, {loss:.4f}\n")

                # --- Save the model checkpoint ---
                # The filename correctly continues from the current iteration number
                checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_iter_{i + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")

            print("\nTraining complete!")

def main():
    # random.seed(66)
    # np.random.seed(66)
    # torch.manual_seed(66)

    print(f"Starting training on device: {Config.DEVICE}")

    # Initialize the model and optimizer
    model = ArmadaNet(ActionManager()).to(Config.DEVICE)
    start_iteration = 0
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Find all checkpoint files
    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('model_iter_') and f.endswith('.pth')]
    
    if checkpoints:
        # Find the checkpoint with the highest iteration number
        latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        
        # Extract the iteration number from the filename
        latest_iter_num = int(latest_checkpoint_file.split('_')[-1].split('.')[0])
        
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint_file)
        print(f"Loading model from the most recent checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        
        # Set the starting iteration for the training loop
        start_iteration = latest_iter_num
        print(f"Resuming training from iteration {start_iteration + 1}")
    else:
        print("No checkpoint found. Starting from scratch.")
        init_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_iter_0.pth")
        torch.save(model.state_dict(), init_checkpoint_path)
        print(f"Randomly initialized model saved to {init_checkpoint_path}")
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # pre_compile JIT geometry functions
    pre_compile_jit_geometry(setup_game())

    # Create the training manager and start the learning process
    alpharmada_trainer = AlphArmada(model, optimizer)
    alpharmada_trainer.learn(start_iteration=start_iteration)

if __name__ == "__main__":
    main()