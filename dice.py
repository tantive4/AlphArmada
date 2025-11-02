import numpy as np

from enum_class import *


RNG = np.random.default_rng()

FULL_DICE_POOL = ((2,2,2), (2,2,2), (2,2,2,2,2))
EMPTY_DICE_POOL = ((0,0,0), (0,0,0), (0,0,0,0,0))

CRIT_INDEX = {Dice.BLACK: 1, Dice.BLUE: 1, Dice.RED: 2}
ACCURACY_INDEX = {Dice.BLUE: 2, Dice.RED: 4}


PROBABILITIES = [np.array(weight) / sum(weight) for weight in ((2, 4, 2), (4, 2, 2), (2, 2, 2, 1, 1))]

ICON_INDICES = {
    Dice.BLACK : ['_','○','○¤'],
    Dice.BLUE : ['○', '¤', '@'],
    Dice.RED : ['_', '○', '¤', '○○', '@']
    }
SHIP_DAMAGE_INDICES = {
    Dice.BLACK: [0, 1, 2],
    Dice.BLUE:  [1, 1, 0],
    Dice.RED:   [0, 1, 1, 2, 1]}

SQUAD_DAMAGE_INDICES = {
    Dice.BLACK: [0, 1, 1],
    Dice.BLUE:  [1, 0, 0],
    Dice.RED:   [0, 1, 0, 2, 0]}

def dice_icon(dice_pool : tuple[tuple[int, ...], ...]) -> dict[Dice, str] :
    icon_dict = {dice_type : ' '.join([(f'{icon} ' * dice_count) for icon, dice_count in zip(ICON_INDICES[dice_type], dice_pool[dice_type.value])]).replace('  ',' ').strip() for dice_type in Dice}
    return {dice_type : dice_pool for dice_type,dice_pool in icon_dict.items() if dice_pool}


def roll_dice(dice_pool: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """
    Simulates rolling Star Wars: Armada dice using NumPy's multinomial distribution.
    """
    return tuple(
        tuple(RNG.multinomial(dice_pool[dice_type], PROBABILITIES[dice_type]).tolist())
        for dice_type in DICE)

def dice_choices(attack_pool_result: tuple[tuple[int, ...], ...], dice_to_modify: int) -> list[tuple[tuple[int, ...], ...]]:
    """
    Generates all possible outcomes of selecting a specific number of dice
    from a larger pool by working directly with the counts of each die face.
    This is more efficient than flattening the list.
    """
    # for 1 dice case
    if dice_to_modify == 1:
        combinations : list[tuple[tuple[int, ...], ...]] = []
        # Iterate through each die color and its face counts
        for color, face_counts in zip(Dice, attack_pool_result):
            # Iterate through each face index and its count
            for face_idx, count in enumerate(face_counts):
                # If there's at least one die of this face, it's a valid choice
                if count > 0:
                    # Create a new, zeroed-out combination dictionary
                    new_combo = (
                        [0] * len(ICON_INDICES[Dice.BLACK]),
                        [0] * len(ICON_INDICES[Dice.BLUE]),
                        [0] * len(ICON_INDICES[Dice.RED])
                    )
                    # Mark the single chosen die in the new combination
                    new_combo[color][face_idx] = 1
                    new_combo_tuple = tuple(tuple(face_list) for face_list in new_combo)
                    combinations.append(new_combo_tuple)
        return combinations
    
    else : raise NotImplementedError("Currently only supports choosing 1 die.")
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
    dice_pool = (800,800,800)

    
    # all_possible_outcomes = generate_all_dice_outcomes(dice_pool)

    print(f"Dice Pool: {dice_pool}")
    # print(f"Total Unique Outcomes: {len(all_possible_outcomes)}\n")

    # # Print each outcome for clarity
    # for i, outcome in enumerate(all_possible_outcomes):
    #     print(f"Outcome {i+1}: {outcome}")

    dice_roll_result = roll_dice(dice_pool)
    print(dice_roll_result)
    # print(f'result : {dice_icon(dice_roll_result)}')
    # for dice_choice in dice_choices(dice_roll_result, 1) :
    #     print(dice_choice)



