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
import json
import math

# Import your custom modules
from armada import Armada
from ship import Ship
from armada_net import ArmadaNet
from game_encoder import encode_game_state
from mcts import MCTS
from action_space import ActionManager
from game_phase import GamePhase, ActionType, get_action_str
from dice import roll_dice

class Config:
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Training Loop
    ITERATIONS = 1 # update model ITERATION times
    SELF_PLAY_GAMES = 1 # generate data for SELF_PLAY_GAMES games during one iteration

    # MCTS
    MCTS_ITERATION = 50
    MAX_GAME_STEP = 1000

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
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def self_play(self) :
        with open('simulation_log.txt', 'w') as f:
            f.write("Game Start\n")
        action_manager = ActionManager()
        memory : list[tuple[GamePhase, dict, np.ndarray]]= []
        game = setup_game()
        mcts_game = copy.deepcopy(game)
        mcts = MCTS(mcts_game, action_manager, self.model, self.config)

        for _ in range(self.config.MAX_GAME_STEP) :
            if game.decision_player is None :
                # chance node case
                if game.phase == GamePhase.SHIP_ATTACK_ROLL_DICE : 
                    if game.attack_info is None :
                        raise ValueError("No attack info for the current game phase.")
                    dice_roll = roll_dice(game.attack_info.dice_to_roll)
                    action = ('roll_dice_action', dice_roll)

                # information set node case
                elif game.phase == GamePhase.SHIP_REVEAL_COMMAND_DIAL :
                    mcts.game.simulation_player = game.current_player
                    if len(mcts.game.get_valid_actions()) != 1 :
                        raise ValueError(f"Multiple valid actions in information set node.\n{mcts.game.get_valid_actions()}")
                    action = mcts.game.get_valid_actions()[0]

            else :
                action_probs = mcts.alpha_mcts_search(game.decision_player)
                memory.append((game.phase, encode_game_state(game), action_probs))
                action = mcts.get_random_best_action(game.decision_player)
                
            print(get_action_str(game, action))
            game.apply_action(action)
            mcts.advance_tree(action, game.get_snapshot())


            if game.winner is not None : 
                return_memory = [(phase, state, action_probs, game.winner) for phase, state, action_probs in memory]
                return return_memory
        raise RuntimeError(f'Maximum simulation steps reached\n{game.get_snapshot()}')
    

    def train(self, memory: list[tuple[GamePhase, dict, np.ndarray, float]]):
        """
        Trains the neural network on a batch of experiences from self-play.
        """
        self.model.train()
        self.optimizer.zero_grad()

        phases, states, target_policies, target_values = zip(*memory)

        # DO NOT convert target_policies to a single tensor here. Keep it as a list of numpy arrays.
        # target_policies = torch.from_numpy(np.array(target_policies)).float().to(self.config.DEVICE) # <- REMOVE THIS LINE

        target_values = torch.tensor(target_values, dtype=torch.float32).to(self.config.DEVICE).view(-1, 1)

        scalar_batch = torch.stack([torch.from_numpy(s['scalar']).float() for s in states]).to(self.config.DEVICE)
        entity_batch = torch.stack([torch.from_numpy(s['entities']).float() for s in states]).to(self.config.DEVICE)
        spatial_batch = torch.stack([torch.from_numpy(s['spatial']).float() for s in states]).to(self.config.DEVICE)
        relation_batch = torch.stack([torch.from_numpy(s['relations']).float() for s in states]).to(self.config.DEVICE)
        phase_batch = list(phases)

        policy_loss = torch.tensor(0.0, device=self.config.DEVICE)
        value_loss = torch.tensor(0.0, device=self.config.DEVICE)

        for i in range(len(memory)):
            policy_logits, value_pred = self.model(
                scalar_batch[i], entity_batch[i], spatial_batch[i],
                relation_batch[i], phase_batch[i]
            )

            # Convert the individual target policy to a tensor INSIDE the loop
            # The shape of target_policies[i] will now correctly match the shape of policy_logits
            current_target_policy = torch.from_numpy(target_policies[i]).float().to(self.config.DEVICE)

            policy_loss += F.cross_entropy(policy_logits.unsqueeze(0), current_target_policy.unsqueeze(0))
            value_loss += F.mse_loss(value_pred, target_values[i])

        # ... (rest of the function remains the same)
        l2_reg = torch.tensor(0., device=self.config.DEVICE)
        for param in self.model.parameters():
            l2_reg += torch.sum(param.pow(2))

        total_loss = (policy_loss + value_loss) / len(memory) + self.config.L2_LAMBDA * l2_reg

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
                print(f"Starting {self_play_iteration+1}th self-play...")
                start_time = time.time()
                new_memory = self.self_play()
                replay_buffer.extend(new_memory)
                end_time = time.time()
                print(f"{self_play_iteration + 1}th Self-play finished in {end_time - start_time:.2f}s. Replay buffer size: {len(replay_buffer)}")

            # --- Training Phase ---
            if len(replay_buffer) < self.config.BATCH_SIZE:
                print("Not enough data in replay buffer to train. Skipping...")
                continue
            
            print("Starting training...")
            start_time = time.time()
            training_batch = random.sample(list(replay_buffer), self.config.BATCH_SIZE)
            loss = self.train(training_batch)
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

def setup_game() -> Armada: 
    with open('ship_info.json', 'r') as f:
        SHIP_DATA: dict[str, dict[str, str | int | list | float]] = json.load(f)
    game = Armada(random.choice([1, -1])) # randomly choose the first player
    
    rebel_ships = (
        Ship(SHIP_DATA['CR90A'], 1), 
        Ship(SHIP_DATA['CR90B'], 1),
        Ship(SHIP_DATA['Neb-B Escort'], 1), 
        Ship(SHIP_DATA['Neb-B Support'], 1))
    
    empire_ships = (
        Ship(SHIP_DATA['VSD1'], -1),
        Ship(SHIP_DATA['VSD2'], -1))

    rebel_deployment = [250, 400, 500, 650]
    empire_deployment = [350, 550]
    random.shuffle(rebel_deployment)
    random.shuffle(empire_deployment)

    yaw_one_click = math.pi/8
    for i, ship in enumerate(rebel_ships) :
        game.deploy_ship(ship, rebel_deployment[i], 175, random.choice([-yaw_one_click, 0, yaw_one_click]), random.randint(1,len(ship.nav_chart)))
    for i, ship in enumerate(empire_ships): 
        game.deploy_ship(ship, empire_deployment[i], 725, math.pi + random.choice([-yaw_one_click, 0, yaw_one_click]), random.randint(1,len(ship.nav_chart)))

    return game

def main():
    # random.seed(66)
    # np.random.seed(66)
    # torch.manual_seed(66)

    config = Config()
    print(f"Starting training on device: {config.DEVICE}")

    # Initialize a single game instance and action manager to pass around
    game = setup_game()
    action_manager = ActionManager()

    # Initialize the model and optimizer
    model = ArmadaNet(action_manager).to(config.DEVICE)
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading model from {config.MODEL_PATH} to continue training.")
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Create the training manager and start the learning process
    alpharmada_trainer = AlphArmada(model, optimizer, config)
    alpharmada_trainer.learn()

if __name__ == "__main__":
    main()