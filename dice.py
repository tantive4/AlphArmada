import random

def roll_dice(dice):
    """
    Simulates rolling Star Wars: Armada dice with specified probabilities.

    Args:
        dice (list): A list [black, blue, red]
                            representing the number of each die type to roll.

    Returns:
        list: A list representing the results in the order:
              [black blank, black hit, black double,
               blue hit, blue critical, blue accuracy,
               red blank, red hit, red critical, red accuracy, red double]
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
    black_weights = [1, 2, 1]

    blue_faces = ["hit", "critical", "accuracy"]
    blue_weights = [2, 1, 1]

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
        results["black_blank"],
        results["black_hit"],
        results["black_double"],
        results["blue_hit"],
        results["blue_critical"],
        results["blue_accuracy"],
        results["red_blank"],
        results["red_hit"],
        results["red_critical"],
        results["red_double"],
        results["red_accuracy"]
    ]

    return output_list

def reroll_dice(dice_result, dice_to_reroll):
    dice_result = [x - y + z for x, y, z in zip(dice_result, dice_to_reroll, roll_dice([sum(dice_to_reroll[:2]), sum(dice_to_reroll[2:4]), sum(dice_to_reroll[4:])]))]
    return dice_result

# input_dice_3 = [10, 2, 2]
# roll_results_3 = roll_dice(input_dice_3)
# print(f"Input: {input_dice_3}")
# print(f"Output: {roll_results_3}")
# print(reroll_dice(roll_results_3, [1,0,0,0,0,0,0,0,0,0,0]))