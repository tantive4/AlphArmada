import os
import random
import shutil
from time import time
from action_phase import Phase
from enum_class import Command
import numpy as np
import csv
from setup_game import setup_game
from game_encoder import encode_game_state
from visualizer import visualize
from enum_class import *


def save_3d_array_with_offset(arr_3d, filename):
    """
    Saves a 3D NumPy array to a CSV file, with each 2D plane
    separated by one blank row.

    Args:
        arr_3d (np.array): The 3D input array (stacked 2D planes).
        filename (str): The name of the output CSV file.
    """
    
    # Get the dimensions: (Planes, Height, Width)
    try:
        num_planes, num_rows, num_cols = arr_3d.shape
    except ValueError as e:
        print(f"Error: Input array is not 3-dimensional. Shape is {arr_3d.shape}")
        print(f"Details: {e}")
        return

    # Create a blank row with the same number of columns
    blank_row = [''] * num_cols

    print(f"Saving array of shape {arr_3d.shape} to {filename}...")

    # Open the file in 'w' (write) mode
    # newline='' is the standard recommendation when working with the csv module
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Loop through each plane (the first dimension)
        for i in range(num_planes):
            
            # --- Add the offset row ---
            # Add a blank row *before* every plane except the very first one
            if i > 0:
                writer.writerow(blank_row)
                
            # --- Write the 2D plane data ---
            # arr_3d[i] is the 2D plane
            # writer.writerows() writes all rows from the 2D array
            writer.writerows(arr_3d[i])

    print(f"Successfully saved to {filename}")



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
    visualize(game, "game_visuals")
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # for _ in range(100):
    #     actions = game.get_valid_actions()
    #     action = actions[0]
    #     game.apply_action(action)
    save_3d_array_with_offset(encode_game_state(game)['spatial'], "game_state_spatial.csv")



    
if __name__ == '__main__':
    main()


# zip -r _game_visual.zip game_visuals
