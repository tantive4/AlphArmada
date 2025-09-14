import json
import os
import math
import shutil

import numpy as np

from armada import Armada, AttackInfo
from ship import Ship, Command, HullSection, _cached_range, _cached_overlapping
from dice import precompute_dice_outcomes 
from game_phase import GamePhase

def main():
    """Main function to set up and run the game."""
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    
    with open('ship_info.json', 'r') as f:
        SHIP_DATA: dict[str, dict[str, str | int | list | float]] = json.load(f)
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")


    game = Armada()

    cr90a = Ship(SHIP_DATA['CR90A'], 1)
    cr90b = Ship(SHIP_DATA['CR90B'], 1)
    neb_escort = Ship(SHIP_DATA['Neb-B Escort'], 1)
    neb_support = Ship(SHIP_DATA['Neb-B Support'], 1)
    victory1 = Ship(SHIP_DATA['VSD1'], -1)
    victory2 = Ship(SHIP_DATA['VSD2'], -1)

    game.deploy_ship(cr90a, 300, 175, 0, 3)
    game.deploy_ship(cr90b, 400, 175, 0, 3)
    game.deploy_ship(neb_escort, 500, 175, 0, 2)
    game.deploy_ship(neb_support, 600, 175, 0, 2)
    game.deploy_ship(victory1, 400, 725, math.pi, 2)
    game.deploy_ship(victory2, 500, 725, math.pi, 2)

    cr90a.asign_command(Command.NAV)
    cr90b.asign_command(Command.NAV)

    neb_escort.asign_command(Command.NAV)
    neb_escort.asign_command(Command.CONFIRE)
    neb_support.asign_command(Command.NAV)
    neb_support.asign_command(Command.CONFIRE)

    victory1.asign_command(Command.REPAIR)
    victory1.asign_command(Command.NAV)
    victory1.asign_command(Command.CONFIRE)
    victory2.asign_command(Command.REPAIR)
    victory2.asign_command(Command.NAV)
    victory2.asign_command(Command.CONFIRE)

    player1 = lambda: game.alpha_mcts_decision(iterations=400)
    player2 = lambda: game.alpha_mcts_decision(iterations=400)


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
