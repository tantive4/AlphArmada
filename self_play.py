from collections import deque
import random
import os
import time
import copy
from tqdm import trange

import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np

from configs import Config
from armada import Armada, setup_game
from cache_function import delete_cache
from jit_geometry import pre_compile_jit_geometry
from armada_net import ArmadaNet
from game_encoder import encode_game_state, get_terminal_value
from para_mcts import MCTS
from action_space import ActionManager
from action_phase import Phase, get_action_str
from dice import roll_dice



class AlphArmada:
    def __init__(self, model : ArmadaNet, optimizer : optim.AdamW) :
        self.model : ArmadaNet = model
        self.optimizer : optim.AdamW = optimizer

    def para_self_play(self) :
        memory : dict[int, list[tuple[Phase, dict, np.ndarray]]] = {para_index : list() for para_index in range(Config.PARALLEL_PLAY)}
        self_play_data : list[tuple[Phase, dict, np.ndarray, float, dict[str, np.ndarray]]] = []

        action_manager = ActionManager()
        para_games : list[Armada] = [setup_game(para_index=para_index) for para_index in range(Config.PARALLEL_PLAY)]
        mcts : MCTS = MCTS(copy.deepcopy(para_games), action_manager, self.model)

        while any(game.winner == 0.0 for game in para_games) :
            simulation_players: dict[int, int] = {i: game.decision_player for i, game in enumerate(para_games) if game.decision_player and game.winner == 0.0}

            if simulation_players :
                # --- Perform MCTS search in parallel for decision nodes ---
                deep_search = random.random() < Config.DEEP_SEARCH_RATIO
                para_action_probs : dict[int, np.ndarray] = mcts.para_search(simulation_players, deep_search=deep_search)
                if deep_search :
                    for para_index in para_action_probs:
                        game : Armada = para_games[para_index]
                        memory[para_index].append((game.phase, encode_game_state(game), para_action_probs[para_index]))
            
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
                if para_index == 0:
                    print(f"Game {para_index+1} Round {game.round} Phase: {game.phase.name}, Action: {get_action_str(game, action)}")
                game.apply_action(action)
                mcts.advance_tree(para_index, action, game.get_snapshot())
                
                # --- Check for terminal states ---
                if game.winner != 0.0:
                    winner, aux_target = get_terminal_value(game)
                    for phase, encoded_state, action_probs in memory[para_index]:
                        self_play_data.append((phase, encoded_state, action_probs, winner, aux_target))
                    memory[para_index].clear()

        return self_play_data
    

    def train(self, memory: list[tuple[Phase, dict, np.ndarray, float]]):
        """
        Trains the neural network on a batch of experiences from self-play.
        """
        if not memory:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        phases, states, target_policies, target_values, aux_targets_list = zip(*memory)
        
        # Prepare batched tensors
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32).to(Config.DEVICE).view(-1, 1)
        scalar_batch = torch.stack([torch.from_numpy(s['scalar']).float() for s in states]).to(Config.DEVICE)
        ship_entity_batch = torch.stack([torch.from_numpy(s['ship_entities']).float() for s in states]).to(Config.DEVICE)
        squad_entity_batch = torch.stack([torch.from_numpy(s['squad_entities']).float() for s in states]).to(Config.DEVICE)
        spatial_batch = torch.stack([torch.from_numpy(s['spatial']).float() for s in states]).to(Config.DEVICE)
        relation_batch = torch.stack([torch.from_numpy(s['relations']).float() for s in states]).to(Config.DEVICE)
        
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
        phase_indices = {}
        for para_index, phase in enumerate(phases):
            if phase.name not in phase_indices:
                phase_indices[phase.name] = []
            phase_indices[phase.name].append(para_index)

        for phase_name, indices in phase_indices.items():
            if not indices:
                continue
            
            # Select predictions and targets for the current group
            group_logits = policy_logits[indices]
            group_target_policies = torch.from_numpy(np.stack([target_policies[para_index] for para_index in indices])).float().to(Config.DEVICE)
            
            # Slice the logits to match the target action space size for this phase
            action_space_size = group_target_policies.shape[1]
            group_logits_sliced = group_logits[:, :action_space_size]
            
            # Use soft-label cross entropy: -sum(p * log_softmax(logits))
            log_probs = F.log_softmax(group_logits_sliced, dim=1)
            policy_loss += -(group_target_policies * log_probs).sum(dim=1).mean()

        # 1. Hull Loss (MSE)
        target_hull = torch.from_numpy(np.stack([t['ship_hulls'] for t in aux_targets_list])).float().to(Config.DEVICE)
        hull_loss = F.mse_loss(model_output["predicted_hull"], target_hull)

        # 2. Squad Loss (BCE)
        target_squads = torch.from_numpy(np.stack([t['squad_hulls'] for t in aux_targets_list])).float().to(Config.DEVICE)
        squad_loss = F.mse_loss(model_output["predicted_squads"], target_squads)

        # 3. Game Length Loss (Cross Entropy)
        target_game_length = torch.from_numpy(np.stack([t['game_length'] for t in aux_targets_list])).float().to(Config.DEVICE)
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
            loss_history = []
            
            # Create directories if they don't exist
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(Config.REPLAY_BUFFER_DIR, exist_ok=True)

            for i in range(start_iteration, Config.ITERATIONS):
                print(f"\n----- Iteration {i+1}/{Config.ITERATIONS} -----\n")
                
                # --- Step 1 & 2: Start fresh self-play and save batches ---
                self.model.eval()
                print("Starting self-play phase...")
                for self_play_iteration in range(Config.SELF_PLAY_GAMES):
                    
                    # self_play() generates a fresh list of experiences for this batch
                    new_memory = self.para_self_play()
                    
                    # Define a unique path for this batch's data
                    # Using a timestamp ensures chronological sorting is easy and robust
                    timestamp = int(time.time() * 1000)
                    replay_buffer_path = os.path.join(
                        Config.REPLAY_BUFFER_DIR,
                        f"replay_{timestamp}_iter_{i}_batch_{self_play_iteration}.pth"
                    )
                    
                    # Save the freshly generated data and clear it from memory
                    torch.save(new_memory, replay_buffer_path)
                
                # --- Step 4: Before training, load all previous self-play data ---
                print("\n--- Preparing for training phase: Loading replay buffers... ---")
                replay_buffer = deque(maxlen=Config.REPLAY_BUFFER_SIZE)

                # Get all saved replay files and sort them from newest to oldest
                all_replay_files = [f for f in os.listdir(Config.REPLAY_BUFFER_DIR) if f.endswith('.pth')]
                all_replay_files.sort(reverse=True)

                for filename in all_replay_files:
                    # Stop if the buffer is full
                    if len(replay_buffer) >= Config.REPLAY_BUFFER_SIZE:
                        break
                    
                    replay_buffer_path = os.path.join(Config.REPLAY_BUFFER_DIR, filename)
                    try:
                        batch_memory = torch.load(replay_buffer_path, weights_only=False)
                        # Extend from the left to add older experiences to the back of the deque
                        replay_buffer.extendleft(batch_memory)
                    except (EOFError, pickle.UnpicklingError):
                        print(f"Warning: Could not load {replay_buffer_path}. File might be corrupted. Skipping.")

                # --- Step 5: Start training on the loaded data ---
                print("Starting training phase...")
                self.model.train()
                for step in trange(Config.EPOCHS):
                    training_batch = random.sample(list(replay_buffer), Config.BATCH_SIZE)
                    loss = self.train(training_batch)
                loss_history.append(loss)
                print(f"Iteration {i+1} completed. Final training loss: {loss:.4f}")

                # --- Save the model checkpoint ---
                # The filename correctly continues from the current iteration number
                checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_iter_{i + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")

            print("\nTraining complete!")
            print(f"Loss history: {loss_history}")

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

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # pre_compile JIT geometry functions
    pre_compile_jit_geometry(setup_game())

    # Create the training manager and start the learning process
    alpharmada_trainer = AlphArmada(model, optimizer)
    alpharmada_trainer.learn(start_iteration=start_iteration)

if __name__ == "__main__":
    main()