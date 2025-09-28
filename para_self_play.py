import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque
import random
import os
import time
import copy

# Import your custom modules
from armada import Armada, setup_game
from ship import _cached_coordinate, _cached_distance, _cached_maneuver_tool, _cached_obstruction, _cached_overlapping, _cached_presence_plane, _cached_range, _cached_threat_plane
from armada_net import ArmadaNet
from game_encoder import encode_game_state
from para_mcts import MCTS
from action_space import ActionManager
from game_phase import GamePhase, ActionType, get_action_str
from dice import roll_dice

class Config:
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Training Loop
    ITERATIONS = 1 # self_play & train for ITERATIONS times
    SELF_PLAY_GAMES = 1 # run SELF_PLAY_GAMES batch self-play games in each iteration
    PARALLEL_PLAY = 2 # run games in batch
    TRAINING_STEPS = 5 # train model TRAINING_STEPS times after each iteration

    # MCTS
    MCTS_ITERATION = 50
    MAX_GAME_STEP = 1000
    TEMPERATURE = 1.0
    EXPLORATION_CONSTANT = 2

    # Replay Buffer
    REPLAY_BUFFER_SIZE = 30000

    # Neural Network Training
    EPOCHS = 1 # learn each data EPOCHS times
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    L2_LAMBDA = 1e-4

    # Model Paths
    MODEL_PATH = "model.pth"
    CHECKPOINT_DIR = "model_checkpoints"

class AlphArmada:
    def __init__(self, model : ArmadaNet, optimizer : optim.AdamW, config : Config) :
        self.model : ArmadaNet = model
        self.optimizer : optim.AdamW = optimizer
        self.config : Config = config

    def self_play(self) :
        memory : dict[int, list[tuple[GamePhase, dict, np.ndarray]]] = {para_index : list() for para_index in range(self.config.PARALLEL_PLAY)}
        self_play_data : list[tuple[GamePhase, dict, np.ndarray, float]] = []

        action_manager = ActionManager()
        para_games : list[Armada] = [setup_game() for _ in range(self.config.PARALLEL_PLAY)]
        mcts : MCTS = MCTS(copy.deepcopy(para_games), action_manager, self.model, self.config)

        while any(game.winner is None for game in para_games) :
            simulation_players: dict[int, int] = {i: game.decision_player for i, game in enumerate(para_games) if game.decision_player is not None and game.winner is None}

            if simulation_players :
                # --- Perform MCTS search in parallel for decision nodes ---
                para_action_probs : dict[int, np.ndarray] = mcts.parallel_search(simulation_players)

                for para_index in para_action_probs:
                    game : Armada = para_games[para_index]
                    memory[para_index].append((game.phase, encode_game_state(game), para_action_probs[para_index]))
            
            # --- Process all games (decision and non-decision) for one step ---
            for para_index in range(self.config.PARALLEL_PLAY) :
                game : Armada = para_games[para_index]
                if game.winner is not None:
                    continue

                if game.decision_player is not None:
                    action = mcts.get_random_best_action(para_index, game.decision_player)

                # Chance Node
                elif game.phase == GamePhase.SHIP_ATTACK_ROLL_DICE:
                    if game.attack_info is None:
                        raise ValueError("No attack info for the current game phase.")
                    dice_roll = roll_dice(game.attack_info.dice_to_roll)
                    action = ('roll_dice_action', dice_roll)

                # Information Set Node
                elif game.phase == GamePhase.SHIP_REVEAL_COMMAND_DIAL:
                    if len(game.get_valid_actions()) != 1:
                        raise ValueError("Multiple valid actions in information set node.")
                    action = game.get_valid_actions()[0]
                print(f"Game {para_index+1} Phase: {game.phase.name}, Action: {get_action_str(game, action)}")
                game.apply_action(action)
                mcts.advance_tree(para_index, action, game.get_snapshot())
                
                # --- Check for terminal states ---
                if game.winner is not None:
                    print(f"Game {para_index+1} ended. Winner: Player {game.winner}")
                    for phase, encoded_state, action_probs in memory[para_index]:
                        self_play_data.append((phase, encoded_state, action_probs, game.winner))
                    memory[para_index].clear()

        _cached_threat_plane.cache_clear()
        _cached_presence_plane.cache_clear()
        _cached_coordinate.cache_clear()
        _cached_distance.cache_clear()
        _cached_maneuver_tool.cache_clear()
        _cached_obstruction.cache_clear()
        _cached_overlapping.cache_clear()
        _cached_range.cache_clear()
        return self_play_data
    

    def train(self, memory: list[tuple[GamePhase, dict, np.ndarray, float]]):
        """
        Trains the neural network on a batch of experiences from self-play.
        """
        if not memory:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        phases, states, target_policies, target_values = zip(*memory)
        
        # Prepare batched tensors
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32).to(self.config.DEVICE).view(-1, 1)
        scalar_batch = torch.stack([torch.from_numpy(s['scalar']).float() for s in states]).to(self.config.DEVICE)
        entity_batch = torch.stack([torch.from_numpy(s['entities']).float() for s in states]).to(self.config.DEVICE)
        spatial_batch = torch.stack([torch.from_numpy(s['spatial']).float() for s in states]).to(self.config.DEVICE)
        relation_batch = torch.stack([torch.from_numpy(s['relations']).float() for s in states]).to(self.config.DEVICE)
        
        # --- Single Batched Forward Pass ---
        policy_logits, value_pred = self.model(
            scalar_batch, entity_batch, spatial_batch,
            relation_batch, list(phases)
        )

        # --- Calculate Losses ---
        
        # Value loss can be calculated on the whole batch at once
        value_loss = F.mse_loss(value_pred, target_values_tensor)

        # Policy loss needs to be calculated per phase group due to different shapes
        policy_loss = torch.tensor(0.0, device=self.config.DEVICE)
        
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
            group_target_policies = torch.from_numpy(np.stack([target_policies[para_index] for para_index in indices])).float().to(self.config.DEVICE)
            
            # Slice the logits to match the target action space size for this phase
            action_space_size = group_target_policies.shape[1]
            group_logits_sliced = group_logits[:, :action_space_size]
            
            # Add to total policy loss
            policy_loss += F.cross_entropy(group_logits_sliced, group_target_policies)

        # L2 Regularization
        l2_reg = torch.tensor(0., device=self.config.DEVICE)
        for param in self.model.parameters():
            l2_reg += torch.sum(param.pow(2))

        # Combine losses
        total_loss = policy_loss + value_loss + self.config.L2_LAMBDA * l2_reg

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
    

    def learn(self):
        """
        The main continuous training loop: self-play -> store -> train.
        """
        replay_buffer = deque(maxlen=self.config.REPLAY_BUFFER_SIZE)
        loss_history = []
        
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

        for i in range(self.config.ITERATIONS):
            print(f"\n----- Iteration {i+1}/{self.config.ITERATIONS} -----")
            
            self.model.eval()
            # --- Self-Play Phase ---
            for self_play_iteration in range(self.config.SELF_PLAY_GAMES):
                print(f"Starting {self_play_iteration+1}th self-play ({self.config.PARALLEL_PLAY} batch games)...")
                start_time = time.time()
                new_memory = self.self_play()
                replay_buffer.extend(new_memory)
                end_time = time.time()
                print(f"{self_play_iteration + 1}th Self-play finished in {end_time - start_time:.2f}s. Replay buffer size: {len(replay_buffer)}")

            # --- Training Phase ---
            if len(replay_buffer) < self.config.BATCH_SIZE:
                print("Not enough data in replay buffer to train. Skipping...")
                continue
            
            start_time = time.time()
            print("Starting training phase...")
            self.model.train()
            # Instead of one batch, loop for a set number of steps
            for step in range(self.config.TRAINING_STEPS):
                # Sample a new mini-batch from the buffer for each step
                training_batch = random.sample(list(replay_buffer), self.config.BATCH_SIZE)
                loss = self.train(training_batch)
                if step % 100 == 0:
                    print(f"  - Step {step}/{self.config.TRAINING_STEPS}, Loss: {loss:.4f}")
                    loss_history.append(loss)
            end_time = time.time()
            print(f"Training finished in {end_time - start_time:.2f}s. Loss: {loss:.4f}")

            # --- Save Checkpoint and update main model file ---
            checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f"model_iter_{i+1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            torch.save(self.model.state_dict(), self.config.MODEL_PATH) # Overwrite the main model
            print(f"Model checkpoint saved to {checkpoint_path}")

        print("\nTraining complete!")
        print(f"Loss history: {loss_history}")



def main():
    # random.seed(66)
    # np.random.seed(66)
    # torch.manual_seed(66)

    config = Config()
    print(f"Starting training on device: {config.DEVICE}")

    # Initialize the model and optimizer
    model = ArmadaNet(ActionManager()).to(config.DEVICE)
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading model from {config.MODEL_PATH} to continue training.")
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Create the training manager and start the learning process
    alpharmada_trainer = AlphArmada(model, optimizer, config)
    alpharmada_trainer.learn()

if __name__ == "__main__":
    main()