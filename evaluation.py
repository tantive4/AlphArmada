import copy
from tqdm import tqdm

from big_deep import BigDeep, load_model
from action_manager import ActionManager
from para_mcts import MCTS
from setup_game import setup_game
from action_phase import Phase
from dice import roll_dice

def evaluation(model1:BigDeep, model2:BigDeep, mcts_iter=100, game_count=128):

    para_games = [setup_game(para_index=i) for i in range(game_count)]

    action_manager = ActionManager()
    mcts1 = MCTS(copy.deepcopy(para_games), action_manager, model1)
    mcts2 = MCTS(copy.deepcopy(para_games), action_manager, model2)

    p1_first = list(range(game_count//2))
    para_games[0].debuging_visual = True

    with tqdm(total=game_count, desc=f"[EVALUATION]", unit="game") as pbar:
        while any(game.winner == 0.0 for game in para_games) :
            para_indices = [i for i, game in enumerate(para_games) if game.decision_player and game.winner == 0.0]
            p1_indices, p2_indices = [], []
            for para_index in para_indices:
                game = para_games[para_index]

                is_first_decision = (game.decision_player == game.first_player)
                is_p1_first = (para_index in p1_first)

                if is_first_decision == is_p1_first:
                    p1_indices.append(para_index)
                else:
                    p2_indices.append(para_index)

            if p1_indices:
                mcts1.para_search(p1_indices, True, manual_iteration=mcts_iter, add_noise=False)
            if p2_indices:
                mcts2.para_search(p2_indices, True, manual_iteration=mcts_iter, add_noise=False)

            for para_index in range(game_count) :
                game = para_games[para_index]
                if game.winner != 0.0:
                    continue
                

                if para_index in p1_indices:
                    action = mcts1.get_best_action(para_index)
                elif para_index in p2_indices:
                    action = mcts2.get_best_action(para_index)
                elif game.phase == Phase.ATTACK_ROLL_DICE:
                    dice_roll = roll_dice(game.attack_info.dice_to_roll)
                    action = ('roll_dice_action', dice_roll)
                else: raise NotImplementedError(f"Unknown Phase\n{game.get_snapshot()}")

                game.apply_action(action)
                mcts1.advance_tree(para_index, action, game.get_snapshot())
                mcts2.advance_tree(para_index, action, game.get_snapshot())

                if game.winner != 0.0:
                    win_player = "model 1" if (game.winner > 0) == (para_index in p1_first) else "model 2"
                    pbar.update(1)
                    pbar.set_postfix(last_winner=win_player)


    raw_win_list = [game.winner for game in para_games]
    p1_win_count = sum(
        1 for para_index, winner in enumerate(raw_win_list) 
        if (winner > 0) == (para_index in p1_first)
        )
    p1_win_sum = sum(
        winner if para_index in p1_first else -winner 
        for para_index, winner in enumerate(raw_win_list)
        )
    print("RAW END DATA")
    print(raw_win_list,"\n\n")
    print(f"Player 1 Win Rate = {p1_win_count/game_count:.2%}")
    print(f"Player 1 Average Score = {p1_win_sum/game_count:.4f}")

def main():
    model1 = load_model(version=256)
    model2 = load_model(version=64)

    model1.eval()
    model2.eval()
    model1.compile_fast_policy()
    model2.compile_fast_policy()

    evaluation(model1, model2)

if __name__ == "__main__":
    main()