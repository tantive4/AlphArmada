import os
import random
import shutil
from time import time

from armada import setup_game
from enum_class import *
def main():
    """Main function to set up and run the game."""
    random.seed(66)
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")


    
    
    # cr90b, neb_b_support, cr90a, neb_b_escort, vsd1, vsd2 = game.ships
    # xwing1, xwing2, xwing3, tie1, tie2, tie3, tie4, tie5, tie6 = game.squads
    
    game = setup_game(debuging_visual=False)
    snapshot = game.get_snapshot()

    start_time = time()
    game.rollout()
    game.revert_snapshot(snapshot)
    end_time = time()
    print(f"First rollout time: {end_time - start_time:.4f} seconds")

    start_time = time()
    for _ in range(10):
        game.rollout()
        game.revert_snapshot(snapshot)
    end_time = time()
    print(f"With JIT: {end_time - start_time:.4f} seconds/10 games")



    
if __name__ == '__main__':
    main()


# zip -r _game_visual.zip game_visuals
