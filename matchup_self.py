import copy
import shutil
import os

from big_deep import load_recent_model
from action_manager import ActionManager
from shared_mcts import MCTS
from setup_game import setup_game
from action_phase import Phase, get_action_str
from dice import roll_dice

def print_separator():
    print("\n" + "="*50 + "\n")

def main():
    # --- HARDCODED SETUP ---
    HUMAN_PLAYER = 1           # 1 for First Player, -1 for Second Player
    MCTS_ITERATIONS = 800      # Number of search iterations for the model's turn
    # -----------------------

    print("Loading Alpharmada Model...")
    model, _ = load_recent_model()
    model.eval()
    model.compile_fast_policy()
    
    action_manager = ActionManager()
    
    print("Setting up randomized game...")
    shutil.rmtree("game_visuals", ignore_errors=True)
    os.makedirs("game_visuals", exist_ok=True)
    game = setup_game()
    
    print("Initializing Shared Tree MCTS...")
    mcts = MCTS(copy.deepcopy(game), action_manager, model)
    print_separator()

    game.debuging_visual = True
    game.visualize("Game Start!")
    print_separator()

    # --- MAIN GAME LOOP ---
    while game.winner == 0.0:

        # 1. Chance Node (Automatic Dice Roll)
        if game.phase == Phase.ATTACK_ROLL_DICE:
            dice_roll = roll_dice(game.attack_info.dice_to_roll)
            action = ('roll_dice_action', dice_roll)
            
            print(f"[DICE] {get_action_str(game, action)}")

        else:
            print(f"\n--- MODEL'S TURN (Phase: {game.phase.name}) ---")
            print(f"Thinking for {MCTS_ITERATIONS} iterations...")
            
            # Run parallel workers on the shared tree
            mcts.shared_search(iteration=MCTS_ITERATIONS)
            
            action = mcts.get_best_action()
            print(f"> Model chose: {get_action_str(game, action)}")
            
        game.apply_action(action)
        mcts.advance_tree(action, game.get_snapshot())

    # --- GAME OVER ---
    print_separator()
    print("GAME OVER")

if __name__ == "__main__":
    main()