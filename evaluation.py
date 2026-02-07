import os
import torch
import random
import copy
import shutil
import numpy as np

# Import game logic and AI modules
from configs import Config
from armada import Armada
from setup_game import setup_game
from big_deep import BigDeep
from action_manager import ActionManager
from para_mcts import MCTS
from action_phase import Phase, get_action_str
from dice import roll_dice
from jit_geometry import pre_compile_jit_geometry
from enum_class import Faction

def load_latest_model(model):
    """Loads the latest checkpoint from the checkpoint directory."""
    if not os.path.exists(Config.CHECKPOINT_DIR):
        print("No checkpoint directory found. Using random weights.")
        return

    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('model_iter_') and f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoint files found. Using random weights.")
        return

    # Find the checkpoint with the highest iteration number
    latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint_file)
    
    print(f"Loading model from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode

def main():
    print(f"Initializing AlphArmada Evaluation on {Config.DEVICE}...")

    # 1. Clear previous visuals
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    os.makedirs("game_visuals", exist_ok=True)

    # 2. Initialize Model
    action_manager = ActionManager()
    model = BigDeep(action_manager).to(Config.DEVICE)
    load_latest_model(model)
    
    # Pre-compile geometry for collision checks
    pre_compile_jit_geometry(setup_game())


    # 3. Setup Game
    # We create a single game with visuals enabled.
    print("Setting up random game scenario...")
    game = setup_game(debuging_visual=True, para_index=0)


    # 4. Initialize MCTS
    # MCTS expects a list of games. We provide a list containing our single game.
    # We pass the game instance directly; MCTS often deepcopies internally, 
    # but we need to keep our 'main' game synchronized with the MCTS tree.
    mcts_game = copy.deepcopy(game)
    mcts_game.debuging_visual = False
    mcts = MCTS([mcts_game], action_manager, model)

    # 5. Select Sides
    human_input = input("Choose your faction (Rebel/Empire) [default: Rebel]: ").strip().lower()
    if human_input == 'empire':
        human_player = Faction.EMPIRE
        ai_player = Faction.REBEL
    else:
        human_player = Faction.REBEL
        ai_player = Faction.EMPIRE

    print(f"\nHuman playing as: {human_player.name}")
    print(f"AI playing as: {ai_player.name}")
    print("Visuals are being saved to ./game_visuals/ folder.\n")

    # 6. Game Loop
    step_count = 0
    while game.winner == 0.0:
        
        # Identify the simulation player (matches self_play logic)
        decision_player = game.decision_player
        
        action = None

        # --- A. Chance Nodes (Dice Rolls) ---
        # The game engine needs external input for dice rolls during chance phases
        if game.phase == Phase.ATTACK_ROLL_DICE:
            print(f"[{game.phase.name}] Rolling Dice...")
            if game.attack_info is None:
                raise ValueError("No attack info for the current game phase.")
            dice_roll = roll_dice(game.attack_info.dice_to_roll)
            action = ('roll_dice_action', dice_roll)
        
        # --- B. Forced/Information Nodes ---
        # e.g., Reveal Command Dial (if only 1 choice or hidden info handling)
        elif game.phase == Phase.SHIP_REVEAL_COMMAND_DIAL:
            # In self_play, decision_player is 0 here.
            # We just take the valid action (reveal the dial).
            valid_actions = game.get_valid_actions()
            action = valid_actions[0]

        elif len(valid_actions := game.get_valid_actions()) == 1:
            action = valid_actions[0]

        # --- C. Human Turn ---
        elif decision_player == human_player:
            # Use the method from armada.pyx that handles CLI input
            action = game.player_decision()

        # --- D. AI Turn ---
        elif decision_player == ai_player:            
            # Run MCTS search
            # We simulate a "batch" of 1 game: {0: ai_player_id}
            # deep_search=True forces more iterations for better inference
            mcts.para_search({0: ai_player.value}, deep_search=True, manual_iteration=50)
            
            # Select the best action based on visit counts (deterministic or stochastic)
            # using get_random_best_action is standard for Alpharmada's inference
            action = mcts.get_best_action(0, ai_player.value)
            
        # --- E. Fallback / Auto Pass ---
        # If the phase implies a decision player but it's not the human or AI (shouldn't happen often)
        # or special phases where decision_player might be 0 but not handled above.
        if action is None:
             print(f"[{game.phase.name}] No decision player or unhandled phase. Auto-passing...")
             valid = game.get_valid_actions()
             if not valid:
                 raise RuntimeError(f"No valid actions in phase {game.phase}")
             action = valid[0]

        # 7. Apply Action and Advance
        game.apply_action(action)
        mcts.advance_tree(0, action, game.get_snapshot())

        step_count += 1
        
        # Safety break
        if step_count > Config.MAX_GAME_STEP:
            print("Maximum game steps reached.")
            break

    # 8. Game Over
    winner_name = "REBEL" if game.winner > 0 else "EMPIRE"
    print(f"\nGame Over! Winner: {winner_name}")
    print(f"Final Score: {game.winner}")

if __name__ == "__main__":
    # Ensure this script runs with access to the package structure
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated by user.")