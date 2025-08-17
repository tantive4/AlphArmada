import random
import itertools
from collections import Counter
from enum import Enum


CRIT_INDICES = [1, 1, 2]
DAMAGE_INDICES = [[0, 1, 2], [1, 1, 0], [0, 1, 1, 2, 0]]

def roll_dice(dice : list[int]) -> list[list[int]]:
    """
    Simulates rolling Star Wars: Armada dice with specified probabilities.

    Args:
        dice (list): A list [black, blue, red]
                            representing the number of each die type to roll.

    Returns:
        list[list[int]]: A list representing the results in the order:
              [[black blank, black hit, black double],
               [blue hit, blue critical, blue accuracy],
               [red blank, red hit, red critical, red accuracy, red double]]
    """
    
    black_dice = dice[0]
    blue_dice = dice[1]
    red_dice = dice[2]

    results = {
        "black_blank": 0,
        "black_hit": 0,
        "black_double": 0,
        "blue_hit": 0,
        "blue_critical": 0,
        "blue_accuracy": 0,
        "red_blank": 0,
        "red_hit": 0,
        "red_critical": 0,
        "red_accuracy": 0,
        "red_double": 0
    }

    black_faces = ["blank", "hit", "double"]
    black_weights = [2, 4, 2]

    blue_faces = ["hit", "critical", "accuracy"]
    blue_weights = [4, 2, 2]

    red_faces = ["blank", "hit", "critical", "accuracy", "double"]
    red_weights = [2, 2, 2, 1, 1]

    for _ in range(black_dice):
        roll = random.choices(black_faces, weights=black_weights, k=1)[0]
        if roll == "blank":
            results["black_blank"] += 1
        elif roll == "hit":
            results["black_hit"] += 1
        elif roll == "double":
            results["black_double"] += 1

    for _ in range(blue_dice):
        roll = random.choices(blue_faces, weights=blue_weights, k=1)[0]
        if roll == "hit":
            results["blue_hit"] += 1
        elif roll == "critical":
            results["blue_critical"] += 1
        elif roll == "accuracy":
            results["blue_accuracy"] += 1

    for _ in range(red_dice):
        roll = random.choices(red_faces, weights=red_weights, k=1)[0]
        if roll == "blank":
            results["red_blank"] += 1
        elif roll == "hit":
            results["red_hit"] += 1
        elif roll == "critical":
            results["red_critical"] += 1
        elif roll == "double":
            results["red_double"] += 1
        elif roll == "accuracy":
            results["red_accuracy"] += 1

    output_list = [
        [results["black_blank"], results["black_hit"], results["black_double"]],
        [results["blue_hit"], results["blue_critical"], results["blue_accuracy"]],
        [results["red_blank"], results["red_hit"], results["red_critical"], results["red_double"], results["red_accuracy"]]
    ]

    return output_list

def reroll_dice(dice_result: list[list[int]], dice_to_reroll: list[list[int]]) -> list[list[int]]:
    """
    Rerolls a specified subset of dice from an initial result.

    Args:
        dice_result (list[list[int]]): The initial dice roll result.
            Format: [[black_faces], [blue_faces], [red_faces]]
        dice_to_reroll (list[list[int]]): The dice to be rerolled from the result.
            The format is the same as dice_result.

    Returns:
        list[list[int]]: The new dice result after the reroll.
    """
    # 1. Determine how many dice of each color to reroll by summing the counts.
    num_to_reroll = [sum(color) for color in dice_to_reroll]

    # 2. Roll only the dice that are being replaced.
    newly_rolled = roll_dice(num_to_reroll)

    # 3. Calculate the final result.
    #    For each face, this is (original count - rerolled count) + new count.
    final_result = []
    for i in range(len(dice_result)):
        color_result = [
            (dice_result[i][j] - dice_to_reroll[i][j]) + newly_rolled[i][j]
            for j in range(len(dice_result[i]))
        ]
        final_result.append(color_result)

    return final_result





def _generate_outcomes_for_color(dice_count: int, num_faces: int) -> list[list[int]]:
    """
    Generates all possible result combinations for a single color of dice.
  
    Args:
        dice_count (int): The number of dice of this color.
        num_faces (int): The number of unique faces on the die.

    Returns:
        List[List[int]]: A list of all possible outcomes for this color.
    """
    if dice_count == 0:
        return [[0] * num_faces]

    face_indices = range(num_faces)
    outcomes = []
    
    # Generate all combinations of face indices with replacement
    for combo in itertools.combinations_with_replacement(face_indices, dice_count):
        # Count how many times each face index appears in the combination
        counts = Counter(combo)
        # Create the result list in the correct order of faces
        result = [counts[i] for i in face_indices]
        outcomes.append(result)
        
    return outcomes

def generate_all_dice_outcomes(dice: list[int]) -> list[list[list[int]]]:
    """
    Creates a list of every possible dice outcome for a given set of dice.

    Args:
        dice (list): A list [black, blue, red]
                     representing the number of each die type.

    Returns:
        list[list[list[int]]]: A list of all unique outcomes. Each outcome
        is formatted like the output of the original `roll_dice` function:
        [[black_faces], [blue_faces], [red_faces]]
    """
    num_black, num_blue, num_red = dice

    # Define the number of unique faces for each die color
    # Black: blank, hit, double
    # Blue: hit, critical, accuracy
    # Red: blank, hit, critical, double, accuracy

    
    # Generate all possible outcomes for each color pile
    black_outcomes = _generate_outcomes_for_color(num_black, len(DAMAGE_INDICES[0]))
    blue_outcomes = _generate_outcomes_for_color(num_blue, len(DAMAGE_INDICES[1]))
    red_outcomes = _generate_outcomes_for_color(num_red, len(DAMAGE_INDICES[2]))
    
    # Combine the outcomes of each color using a Cartesian product
    # This pairs every black outcome with every blue outcome and every red outcome.
    all_combinations = itertools.product(black_outcomes, blue_outcomes, red_outcomes)
    
    # Format the final list
    final_outcomes = [[black, blue, red] for black, blue, red in all_combinations]
    
    return final_outcomes


if __name__ == "__main__":
    # --- Example Usage ---
    # Input: 1 black die, 1 blue die, 0 red dice
    dice_pool = [1, 1, 0]
    all_possible_outcomes = generate_all_dice_outcomes(dice_pool)

    print(f"Dice Pool: {dice_pool}")
    print(f"Total Unique Outcomes: {len(all_possible_outcomes)}\n")

    # Print each outcome for clarity
    for i, outcome in enumerate(all_possible_outcomes):
        print(f"Outcome {i+1}: {outcome}")


class Critical(Enum) :
    STANDARD = 0