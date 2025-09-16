import json
import os
import math
import shutil

import numpy as np

from armada import Armada, AttackInfo
from ship import Ship, Command, HullSection, _cached_range, _cached_overlapping
from game_phase import GamePhase

def main():
    """Main function to set up and run the game."""
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    
    with open('ship_info.json', 'r') as f:
        SHIP_DATA: dict[str, dict[str, str | int | list | float]] = json.load(f)
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")


    game = Armada(initiative=1)

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

    game.rollout()

if __name__ == '__main__':
    main()



# to see the time spent
# python -m cProfile -o profile_results self_play.py

# to see the result
# python -m pstats profile_results
# % sort cumtime
# % stats 50

# zip -r game_visual.zip game_visuals
