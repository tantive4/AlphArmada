import random
import itertools
from collections import Counter
from enum import Enum

# Black = [blank, hit, double]
# Blue = [hit, critical, accuracy]
# Red = [blank, hit, critical, double, accuracy]



class Dice(Enum) :
    BLACK = 0
    BLUE = 1
    RED = 2
    def __str__(self):
        return self.name
    __repr__ = __str__
    
CRIT_INDEX = {Dice.BLACK: 1, Dice.BLUE: 1, Dice.RED: 2}
ACCURACY_INDEX = {Dice.BLUE: 2, Dice.RED: 4}
ICON_INDICES = {
    Dice.BLACK : ['_','○','○¤'],
    Dice.BLUE : ['○', '¤', '@'],
    Dice.RED : ['_', '○', '¤', '○○', '@']
}
DAMAGE_INDICES = {
    Dice.BLACK: [0, 1, 2],
    Dice.BLUE:  [1, 1, 0],
    Dice.RED:   [0, 1, 1, 2, 1]
}
def dice_icon(dice_pool : dict[Dice, list[int]]) -> dict[Dice, str] :
    return {dice_type : ' '.join([(f'{icon} ' * dice_count) for icon, dice_count in zip(ICON_INDICES[dice_type], dice_pool[dice_type])]).replace('  ',' ').strip() for dice_type in Dice}

def roll_dice(dice_pool : dict[Dice, int]) -> dict[Dice, list[int]]:
    """
    Simulates rolling Star Wars: Armada dice with specified probabilities.

    Args:
        dice (list): A list [black, blue, red]
                            representing the number of each die type to roll.

    Returns:
        dict: A list representing the results in the order:
              {BLACK : [black blank, black hit, black double],
               BLUE : [blue hit, blue critical, blue accuracy],
               RED : [red blank, red hit, red critical, red double, red accuracy]}
    """
    
    black_dice = dice_pool.get(Dice.BLACK,0)
    blue_dice = dice_pool.get(Dice.BLUE,0)
    red_dice = dice_pool.get(Dice.RED,0)

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
        "red_double": 0,
        "red_accuracy": 0
    }

    black_faces = ["blank", "hit", "double"]
    black_weights = [2, 4, 2]

    blue_faces = ["hit", "critical", "accuracy"]
    blue_weights = [4, 2, 2]

    red_faces = ["blank", "hit", "critical", "double", "accuracy"]
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
            results["red_accuracy"] += 1
        elif roll == "accuracy":
            results["red_double"] += 1

    dice_result = {
        Dice.BLACK : [results["black_blank"], results["black_hit"], results["black_double"]],
        Dice.BLUE : [results["blue_hit"], results["blue_critical"], results["blue_accuracy"]],
        Dice.RED : [results["red_blank"], results["red_hit"], results["red_critical"], results["red_double"], results["red_accuracy"]]
    }

    return dice_result







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

def generate_all_dice_outcomes(dice_pool: dict[Dice,int]) -> list[dict[Dice, list[int]]]:
    """
    Creates a list of every possible dice outcome for a given set of dice.

    Args:
        dice (list): A list [black, blue, red]
                     representing the number of each die type.

    Returns:
        list[dict]: A list of all unique outcomes. Each outcome
        is formatted like the output of the original **roll_dice** function:
        {Dice.BLACK : [black_faces], Dice.BLUE : [blue_faces], Dice.RED : [red_faces]}
    """
    black_outcomes = _generate_outcomes_for_color(dice_pool[Dice.BLACK], 3)
    blue_outcomes = _generate_outcomes_for_color(dice_pool[Dice.BLUE], 3)
    red_outcomes = _generate_outcomes_for_color(dice_pool[Dice.RED], 5)
    
    # Combine the outcomes of each color using a Cartesian product
    # This pairs every black outcome with every blue outcome and every red outcome.
    all_combinations = itertools.product(black_outcomes, blue_outcomes, red_outcomes)
    
    # Format the final list
    final_outcomes = [
        {Dice.BLACK: black, Dice.BLUE: blue, Dice.RED: red}
        for black, blue, red in all_combinations
    ]
    
    return final_outcomes

def dice_choice_combinations(attack_pool_result: dict[Dice, list[int]], dice_to_modify: int) -> list[dict[Dice, list[int]]]:
    """
    Generates all possible outcomes of selecting a specific number of dice
    from a larger pool by working directly with the counts of each die face.
    This is more efficient than flattening the list.
    """
    
    # We'll build the combinations using a recursive helper function
    combinations = []
    
    def find_combos_recursive(current_combo : dict[Dice, list[int]], dice_left : int, start_color : Dice | None, start_face_idx : int):
        if dice_left == 0:
            combinations.append({k: v[:] for k, v in current_combo.items()})
            return

        if start_color is None:
            return

        # Determine the next position to check
        next_face_idx = start_face_idx + 1
        next_color = start_color
        if next_face_idx >= len(attack_pool_result[start_color]):
            next_face_idx = 0
            # Move to the next color in the enum's defined order
            next_color = Dice(start_color.value + 1) if start_color.value + 1 < len(Dice) else None

        # Option 1: Skip the current die face and move to the next.
        find_combos_recursive(current_combo, dice_left, next_color, next_face_idx)
        
        # Option 2: Take one or more dice of the current face type.
        available_count = attack_pool_result[start_color][start_face_idx]
        
        for i in range(1, min(dice_left, available_count) + 1):
            current_combo[start_color][start_face_idx] += i
            find_combos_recursive(current_combo, dice_left - i, next_color, next_face_idx)
            current_combo[start_color][start_face_idx] -= i
        

    # Kick off the recursion
    initial_combo = {
        Dice.BLACK: [0, 0, 0],
        Dice.BLUE:  [0, 0, 0],
        Dice.RED:   [0, 0, 0, 0, 0]
    }
    find_combos_recursive(initial_combo, dice_to_modify, Dice.BLACK, 0)
    
    return combinations

if __name__ == "__main__":
    # --- Example Usage ---
    # Input: 1 black die, 1 blue die, 0 red dice
    dice_pool = {Dice.BLACK : 5, Dice.BLUE : 5, Dice.RED : 5}

    
    # all_possible_outcomes = generate_all_dice_outcomes(dice_pool)

    print(f"Dice Pool: {dice_pool}")
    # print(f"Total Unique Outcomes: {len(all_possible_outcomes)}\n")

    # # Print each outcome for clarity
    # for i, outcome in enumerate(all_possible_outcomes):
    #     print(f"Outcome {i+1}: {outcome}")

    dice_roll_result = roll_dice(dice_pool)
    print(dice_roll_result)
    print(f'result : {dice_icon(dice_roll_result)}')
    # for dice_choice in dice_choice_combinations(dice_roll_result, 2) :
    #     print(dice_choice)

class Critical(Enum) :
    STANDARD = 0
    def __str__(self) :
        return self.name
    __repr__ = __str__