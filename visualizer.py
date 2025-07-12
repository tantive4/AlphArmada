# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import os

def visualize_board(ships, player_edge, short_edge, step_number):
    """
    Saves a visualization of the game board to a file.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, player_edge)
    ax.set_ylim(0, short_edge)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Game State - Step {step_number}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.invert_yaxis() # Invert y-axis for a top-down view

    for ship in ships:
        if not ship.destroyed:
            width, height = ship.size_dimension
            rect = patches.Rectangle((-width/2, -height), width, height, linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.6)
            transform = plt.matplotlib.transforms.Affine2D().rotate_around(0, 0, ship.orientation) + \
                        plt.matplotlib.transforms.Affine2D().translate(ship.x, ship.y) + \
                        ax.transData
            rect.set_transform(transform)
            ax.add_patch(rect)
            plt.plot(ship.x, ship.y, 'ro', markersize=4)
            plt.text(ship.x, ship.y - 20, ship.name, fontsize=9, ha='center', color='black')

    # Create a directory to store visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # Save the plot to a file instead of showing it
    filename = f'visualizations/game_state_{step_number:03d}.png'
    plt.savefig(filename)
    plt.close(fig) # Close the plot figure to free up memory
    print(f"Visualization saved to {filename}")