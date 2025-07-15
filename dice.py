import random
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

# input_dice_3 = [10, 2, 2]
# roll_results_3 = roll_dice(input_dice_3)
# print(f"Input: {input_dice_3}")
# print(f"Output: {roll_results_3}")
# print(reroll_dice(roll_results_3, [[1,0,0],[0,0,0],[0,0,0,0,0]]))