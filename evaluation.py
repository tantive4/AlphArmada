import copy

from big_deep import BigDeep, load_model
from action_manager import ActionManager
from para_mcts import MCTS
from setup_game import setup_game
from action_phase import Phase
from dice import roll_dice

def evaluation(model1:BigDeep, model2:BigDeep, mcts_iter=100, game_count=128):

    para_games = [setup_game(para_index=i) for i in range(game_count)]
    p1_first = list(range(60))

    action_manager = ActionManager()
    
    mcts1 = MCTS(copy.deepcopy(para_games), action_manager, model1)
    mcts2 = MCTS(copy.deepcopy(para_games), action_manager, model2)

    while any(game.winner == 0.0 for game in para_games) :
        para_indices = [i for i, game in enumerate(para_games) if game.decision_player and game.winner == 0.0]
        if para_indices:
            mcts1.para_search(para_indices, manual_iteration=mcts_iter, add_noise=False)
            mcts2.para_search(para_indices, manual_iteration=mcts_iter, add_noise=False)

        for para_index in range(game_count) :
            game = para_games[para_index]
            if game.winner != 0.0:
                continue
            
            is_first_player = (game.decision_player == game.first_player)
            is_p1_index = (para_index in p1_first)
            if game.phase == Phase.ATTACK_ROLL_DICE:
                dice_roll = roll_dice(game.attack_info.dice_to_roll)
                action = ('roll_dice_action', dice_roll)
            elif is_first_player == is_p1_index:
                action = mcts1.get_best_action(para_index)
            else:
                action = mcts2.get_best_action(para_index)


            game.apply_action(action)
            mcts1.advance_tree(para_index, action, game.get_snapshot())
            mcts2.advance_tree(para_index, action, game.get_snapshot())
        
    raw_win_list = [game.winner for game in para_games]
    p1_win_count = sum(
        1 for para_index, winner in enumerate(raw_win_list) 
        if (winner > 0) == (para_index in p1_first)
        )
    p1_win_sum = sum(
        winner if para_index in p1_first else -winner 
        for para_index, winner in enumerate(raw_win_list)
        )
    print(f"Player 1 Win Rate = {p1_win_count/game_count:.2%}")
    print(f"Player 1 Average Score = {p1_win_sum/game_count:.4f}")

def main():
    model1 = load_model(version=100)
    model2 = load_model(version=0)
    evaluation(model1, model2)

if __name__ == "__main__":
    main()