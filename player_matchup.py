import copy
from big_deep import load_recent_model, BigDeep
from action_manager import ActionManager
from shared_mcts import MCTS
from setup_game import setup_game
from action_phase import Phase
from dice import roll_dice

def print_separator():
    print("\n" + "="*50 + "\n")

def main():
    # --- HARDCODED SETUP ---
    HUMAN_PLAYER = 1           # 1 for First Player, -1 for Second Player
    MCTS_ITERATIONS = 200      # Number of search iterations for the model's turn
    # -----------------------

    print("Loading Alpharmada Model...")
    model, _ = load_recent_model()
    model.eval()
    model.compile_fast_policy()
    
    action_manager = ActionManager()
    
    print("Setting up randomized game...")
    game = setup_game()
    
    print("Initializing Shared Tree MCTS...")
    mcts = MCTS(copy.deepcopy(game), action_manager, model)

    print_separator()
    print(f"Game Start! You are Player {'1 (First)' if HUMAN_PLAYER == 1 else '2 (Second)'}.")
    game.debugging_visual = True
    game.visualize("Game Start!")
    print_separator()

    # --- MAIN GAME LOOP ---
    while game.winner == 0.0:
        
        # 1. Chance Node (Automatic Dice Roll)
        if game.phase == Phase.ATTACK_ROLL_DICE:
            dice_roll = roll_dice(game.attack_info.dice_to_roll)
            action = ('roll_dice_action', dice_roll)
            
            print(f"[SYSTEM] Dice Rolled: {dice_roll}")

        # 2. Human Turn
        elif game.decision_player == HUMAN_PLAYER:
            print(f"\n--- YOUR TURN (Phase: {game.phase.name}) ---")
            valid_actions = game.get_valid_actions()
            
            # Display actions
            for i, action in enumerate(valid_actions, start=1):
                print(f"  [{i}] {action}")
                
            # Input loop
            while True:
                try:
                    user_input = input(f"\nSelect an action [1-{len(valid_actions)}]: ")
                    action_index = int(user_input.strip()) - 1
                    
                    if 0 <= action_index < len(valid_actions):
                        action = valid_actions[action_index]
                        
                        # Ask for confirmation
                        confirm = input(f"> You chose: {action}. Confirm? (y/n): ").strip().lower()
                        
                        if confirm == 'y':
                            break  # Exit the loop successfully
                        else:
                            print("Selection cancelled. Let's try again.")
                            
                    else:
                        print("Invalid index. Out of range.")
                        
                except ValueError:
                    print("Invalid input. Please enter an integer.")

            print(f"> You chose: {action}")

        # 3. Model Turn
        else:
            print(f"\n--- MODEL'S TURN (Phase: {game.phase.name}) ---")
            print(f"Thinking for {MCTS_ITERATIONS} iterations...")
            
            # Run parallel workers on the shared tree
            mcts.shared_search(iteration=MCTS_ITERATIONS)
            
            action = mcts.get_best_action()
            print(f"> Model chose: {action}")
            
        game.apply_action(action)
        mcts.advance_tree(action, game.get_snapshot())

    # --- GAME OVER ---
    print_separator()
    print("GAME OVER")
    if (game.winner>0) == (HUMAN_PLAYER>0):
        print("Congratulations, you won!")
    else:
        print("The model defeated you.")

if __name__ == "__main__":
    main()