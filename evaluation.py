import copy
from tqdm import tqdm
import argparse

from storage_manager import download_model_version
from big_deep import BigDeep, load_model
from action_manager import ActionManager
from para_mcts import MCTS
from setup_game import setup_game
from action_phase import Phase
from dice import roll_dice

def evaluation(model1:BigDeep, model2:BigDeep, mcts_iter=200, game_count=128):

    games_half = [setup_game() for _ in range(game_count//2)]
    games_half2 = copy.deepcopy(games_half)
    para_games = [*games_half, *games_half2]
    for para_index, para_game in enumerate(para_games):
        para_game.para_index = para_index


    action_manager = ActionManager()
    mcts1 = MCTS(copy.deepcopy(para_games), action_manager, model1)
    mcts2 = MCTS(copy.deepcopy(para_games), action_manager, model2)

    p1_first = list(range(game_count//2))
    # para_games[0].debuging_visual = True

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
    draw = [game.get_point(1) == 0 and game.get_point(-1) == 0 for game in para_games]
    p1_win_count = sum(
        1 for para_index, winner in enumerate(raw_win_list) 
        if (winner > 0) == (para_index in p1_first)
        )
    print("RAW END DATA")
    print(raw_win_list,"\n\n")
    print(draw, "\n\n")
    print(f"Player 1 Win Rate = {p1_win_count/game_count:.2%}")
    print(f"Draw Rate = {sum(draw)/game_count:.4f}")

def ready_model(version:int):
    download_model_version(version)
    model = load_model(version=version)
    model.eval()
    model.compile_fast_policy()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs=2, type=int, required=True, help="Two model versions to compare (e.g., --versions 256 64)")    
    args = parser.parse_args()
    versions = args.versions

    model1 = ready_model(versions[0])
    model2 = ready_model(versions[1])

    print(f"Starting evaluation between version_{versions[0]}(P1) and version_{versions[1]}(P2)...\n")

    evaluation(model1, model2)

if __name__ == "__main__":
    main()