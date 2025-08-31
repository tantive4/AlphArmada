from armada import Armada, AttackInfo
from ship import Ship, Command, HullSection, _cached_range, _cached_overlapping
import json
import os
import math
import copy
import shutil
import multiprocessing
import random
from dice import precompute_dice_outcomes 
from game_phase import GamePhase

def main():
    """Main function to set up and run the game."""
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    random.seed(66)
    with open('ship_info.json', 'r') as f:
        SHIP_DATA: dict[str, dict[str, str | int | list | float]] = json.load(f)
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")

    game = Armada()

    cr90 = Ship(SHIP_DATA['CR90A'], 1)
    nebulon = Ship(SHIP_DATA['Neb-B Escort'], 1)
    victory = Ship(SHIP_DATA['VSD2'], -1)

    game.deploy_ship(cr90, 300, 175, 0, 2)
    game.deploy_ship(nebulon, 600, 175, 0, 2)
    game.deploy_ship(victory, 450, 725, math.pi, 2)

    cr90.asign_command(Command.REPAIR)
    nebulon.asign_command(Command.NAV)
    nebulon.asign_command(Command.CONFIRE)
    victory.asign_command(Command.REPAIR)
    victory.asign_command(Command.NAV)
    victory.asign_command(Command.CONFIRE)


    # for MULTI CORE simulation
    # CPU_CORE = 4
    # player1 = lambda: game.mcts_decision_parallel(iterations=1600, num_processes=CPU_CORE)
    # player2 = lambda: game.mcts_decision_parallel(iterations=800, num_processes=CPU_CORE)

    # for SINGLE CORE simulation
    player1 = lambda: game.mcts_decision(iterations=400)
    player2 = lambda: game.mcts_decision(iterations=400)


    # for RANDOM PLAYER simulation
    # player1 = game.random_decision
    # player2 = game.random_decision

    game.play(player1, player2)




if __name__ == '__main__':
    precompute_dice_outcomes()
    main()



# to see the time spent
# python -m cProfile -o profile_results game.py

# to see the result
# python -m pstats profile_results
# % sort cumtime
# % stats 50

# zip -r game_visual.zip game_visuals
