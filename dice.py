import random
import itertools
from collections import Counter
from enum import Enum, IntEnum

# --- Global Cache ---
# This dictionary will store the outcomes for a given number and color of dice
# to avoid re-calculating them. The key will be a tuple like (Dice.BLACK, 3).
_DICE_OUTCOME_CACHE = {}


class Dice(IntEnum) :
    BLACK = 0
    BLUE = 1
    RED = 2
    def __str__(self):
        return self.name
    __repr__ = __str__

FULL_DICE_POOL = {Dice.BLACK : [2,2,2], Dice.BLUE : [2,2,2], Dice.RED : [2,2,2,2,2]}


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
    icon_dict = {dice_type : ' '.join([(f'{icon} ' * dice_count) for icon, dice_count in zip(ICON_INDICES[dice_type], dice_pool[dice_type])]).replace('  ',' ').strip() for dice_type in Dice}
    return {dice_type : dice_pool for dice_type,dice_pool in icon_dict.items() if dice_pool}

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
        Dice.BLACK : (results["black_blank"], results["black_hit"], results["black_double"]),
        Dice.BLUE : (results["blue_hit"], results["blue_critical"], results["blue_accuracy"]),
        Dice.RED : (results["red_blank"], results["red_hit"], results["red_critical"], results["red_double"], results["red_accuracy"])
    }

    return dice_result


def _generate_outcomes_for_color(dice_count: int, num_faces: int, dice_type: Dice) -> list[list[int]]:
    """
    Generates all possible result combinations for a single color of dice.
    This function is used by the pre-computation step.
    """
    if dice_count == 0:
        return [[0] * num_faces]

    # Create a unique key for the cache
    cache_key = (dice_type, dice_count)
    
    # Check if the result is already in the cache (it will be after pre-computation)
    if cache_key in _DICE_OUTCOME_CACHE:
        return _DICE_OUTCOME_CACHE[cache_key]

    # If not in cache (e.g., for > max_dice), calculate the outcomes on the fly
    face_indices = range(num_faces)
    outcomes = []
    
    for combo in itertools.combinations_with_replacement(face_indices, dice_count):
        counts = Counter(combo)
        result = [counts[i] for i in face_indices]
        outcomes.append(result)
    
    # Store the result in the cache for future use
    _DICE_OUTCOME_CACHE[cache_key] = outcomes
    return outcomes

def precompute_dice_outcomes(max_dice: int = 8):
    """
    Calculates and caches all dice outcomes up to max_dice for each color.
    This is run once when the module is imported for maximum performance.
    """
    for dice_type in Dice:
        num_faces = len(ICON_INDICES[dice_type])
        for i in range(1, max_dice + 1):
            # This call will compute and store the results in the global cache
            _generate_outcomes_for_color(i, num_faces, dice_type)

def generate_all_dice_outcomes(dice_pool: dict[Dice,int]) -> list[dict[Dice, list[int]]]:
    """
    Creates a list of every possible dice outcome for a given set of dice
    by combining pre-calculated outcomes for each color from the cache.
    """
    # Get outcomes for each color (this will use the cache)
    black_outcomes = _generate_outcomes_for_color(dice_pool.get(Dice.BLACK, 0), 3, Dice.BLACK)
    blue_outcomes = _generate_outcomes_for_color(dice_pool.get(Dice.BLUE, 0), 3, Dice.BLUE)
    red_outcomes = _generate_outcomes_for_color(dice_pool.get(Dice.RED, 0), 5, Dice.RED)
    
    # Combine the outcomes of each color using a Cartesian product
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
    # for 1 dice case
    if dice_to_modify == 1:
        combinations : list[dict[Dice, list[int]]]= []
        # Iterate through each die color and its face counts
        for color, face_counts in attack_pool_result.items():
            # Iterate through each face index and its count
            for face_idx, count in enumerate(face_counts):
                # If there's at least one die of this face, it's a valid choice
                if count > 0:
                    # Create a new, zeroed-out combination dictionary
                    new_combo = {
                        Dice.BLACK: [0] * len(attack_pool_result.get(Dice.BLACK, [])),
                        Dice.BLUE:  [0] * len(attack_pool_result.get(Dice.BLUE, [])),
                        Dice.RED:   [0] * len(attack_pool_result.get(Dice.RED, []))
                    }
                    # Mark the single chosen die in the new combination
                    new_combo[color][face_idx] = 1
                    combinations.append(new_combo)
        return combinations
    
    # We'll build the combinations using a recursive helper function
    combinations = []
    
    # We use a list of colors to iterate more robustly than relying on enum values.
    colors_to_check = sorted(attack_pool_result.keys(), key=lambda d: d.value)
    
    def find_combos_recursive(current_combo: dict[Dice, list[int]], dice_left: int, color_idx: int, face_idx: int):
        # Base case: A valid combination of the required size has been found.
        if dice_left == 0:
            combinations.append({k: v[:] for k, v in current_combo.items()})
            return

        # Base case: We have run out of dice faces to check.
        if color_idx >= len(colors_to_check):
            return

        # Determine the current color and details for this recursive step.
        current_color = colors_to_check[color_idx]
        num_faces = len(attack_pool_result[current_color])

        # Determine the next position to check for the recursive calls.
        next_face_idx = face_idx + 1
        next_color_idx = color_idx
        if next_face_idx >= num_faces:
            next_face_idx = 0
            next_color_idx += 1

        # Option 1: Skip the current die face and move to the next.
        find_combos_recursive(current_combo, dice_left, next_color_idx, next_face_idx)
        
        # Option 2: Take one or more dice of the current face type.
        available_count = attack_pool_result[current_color][face_idx]
        if available_count > 0:
            for i in range(1, min(dice_left, available_count) + 1):
                current_combo[current_color][face_idx] += i
                find_combos_recursive(current_combo, dice_left - i, next_color_idx, next_face_idx)
                # Backtrack: undo the change for the next iteration.
                current_combo[current_color][face_idx] -= i
    
    # Kick off the recursion with an empty starting combination.
    initial_combo = {color: [0] * len(counts) for color, counts in attack_pool_result.items()}
    find_combos_recursive(initial_combo, dice_to_modify, 0, 0)
    
    return combinations


if __name__ == "__main__":
    # --- Example Usage ---
    # Input: 1 black die, 1 blue die, 0 red dice
    dice_pool = {Dice.BLACK : 2, Dice.BLUE : 2, Dice.RED : 2}

    
    # all_possible_outcomes = generate_all_dice_outcomes(dice_pool)

    print(f"Dice Pool: {dice_pool}")
    # print(f"Total Unique Outcomes: {len(all_possible_outcomes)}\n")

    # # Print each outcome for clarity
    # for i, outcome in enumerate(all_possible_outcomes):
    #     print(f"Outcome {i+1}: {outcome}")

    dice_roll_result = roll_dice(dice_pool)
    print(dice_roll_result)
    print(f'result : {dice_icon(dice_roll_result)}')
    for dice_choice in dice_choice_combinations(dice_roll_result, 2) :
        print(dice_choice)

class Critical(Enum) :
    STANDARD = 0
    def __str__(self) :
        return self.name
    __repr__ = __str__

