import os
import random
import shutil

from armada import setup_game

def main():
    """Main function to set up and run the game."""
    random.seed(66)
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")


    game = setup_game(debuging_visual=False)
    game.rollout()


    
if __name__ == '__main__':
    main()



# to see the time spent
# python -m cProfile -o profile_results sample_rollout.py

# to see the result
# python -m pstats profile_results
# % sort cumtime
# % stats 50

# zip -r game_visual.zip game_visuals
