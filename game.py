from armada import Armada
from ship import Ship, Command, _cached_measurements, HullSection
import json
import os
import math
import copy
import shutil
import multiprocessing
from dice import precompute_dice_outcomes  # <-- Add this import

def main():
    """Main function to set up and run the game."""
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")

    with open('ship_info.json', 'r') as f:
        SHIP_DATA: dict[str, dict[str, str | int | list | float]] = json.load(f)
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")

    game = Armada()

    cr90 = Ship(SHIP_DATA['CR90A'], 1)
    nebulon = Ship(SHIP_DATA['Neb-B Escort'], 1)
    victory = Ship(SHIP_DATA['VSD2'], -1)

    game.deploy_ship(cr90, 600, 175, math.pi/4, 2)
    game.deploy_ship(nebulon, 300, 175, 0, 2)
    game.deploy_ship(victory, 450, 725, math.pi, 2)

    cr90.asign_command(Command.NAVIGATION)
    nebulon.asign_command(Command.NAVIGATION)
    nebulon.asign_command(Command.CONCENTRATE_FIRE)
    victory.asign_command(Command.NAVIGATION)
    victory.asign_command(Command.CONCENTRATE_FIRE)
    victory.asign_command(Command.CONCENTRATE_FIRE)


    # for MULTI CORE simulation
    # CPU_CORE = 4
    # player1 = lambda: game.mcts_decision_parallel(iterations=1600, num_processes=CPU_CORE)
    # player2 = lambda: game.mcts_decision_parallel(iterations=800, num_processes=CPU_CORE)

    # for SINGLE CORE simulation
    player1 = lambda: game.mcts_decision(iterations=100)
    player2 = lambda: game.mcts_decision(iterations=100)


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