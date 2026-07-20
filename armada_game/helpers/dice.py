import itertools

import numpy as np

from armada_game.helpers.enum_class import *


RNG = np.random.default_rng()

FULL_DICE_POOL = ((2,2,2), (2,2,2), (2,2,2,2,2))
EMPTY_DICE_POOL = ((0,0,0), (0,0,0), (0,0,0,0,0))
DICE_CHOICE_1 = ((1,0,0),
                 (0,1,0),
                 (0,0,1))

CRIT_INDEX = {Dice.BLACK: 1, Dice.BLUE: 1, Dice.RED: 2}
ACCURACY_INDEX = {Dice.BLUE: 2, Dice.RED: 4}
ACCURACY_DICE_1 = ((0,0,0),
                   (0,0,1),
                   (0,0,0,0,1))

PROBABILITIES = [np.array(weight) / sum(weight) for weight in ((2, 4, 2), (4, 2, 2), (2, 2, 2, 1, 1))]

ICON_INDICES = (
    ["blank", "hit", "hit_crit"],
    ["hit", "crit", "accuracy"],
    ["blank", "hit", "crit", "double_hit", "accuracy"],
)
SHIP_DAMAGE_INDICES = (
    [0, 1, 2],
    [1, 1, 0],
    [0, 1, 1, 2, 1]
)

SQUAD_DAMAGE_INDICES = (
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0, 2, 0]
)

def dice_icon(dice_pool : tuple[tuple[int, ...], ...]) -> dict[int, str] :
    icon_dict = {dice_type : ' '.join([(f'{icon} ' * dice_count) 
                                       for icon, dice_count in zip(ICON_INDICES[dice_type], dice_pool[dice_type])]).replace('  ',' ').strip() 
                                       for dice_type in DICE}
    return {Dice(dice_type) : dice_pool for dice_type,dice_pool in icon_dict.items() if dice_pool}


def roll_dice(dice_pool: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """
    Simulates rolling Star Wars: Armada dice using NumPy's multinomial distribution.
    """
    black_roll = tuple(RNG.multinomial(dice_pool[Dice.BLACK], PROBABILITIES[Dice.BLACK]).tolist())
    blue_roll = tuple(RNG.multinomial(dice_pool[Dice.BLUE], PROBABILITIES[Dice.BLUE]).tolist())
    red_roll = tuple(RNG.multinomial(dice_pool[Dice.RED], PROBABILITIES[Dice.RED]).tolist())

    return (black_roll, blue_roll, red_roll)

def dice_single_choice(attack_pool_result: tuple[tuple[int, ...], ...]) -> list[tuple[tuple[int, ...], ...]]:
    """
    Generates all possible outcomes of selecting exactly one die from the pool,
    working directly with the counts of each die face rather than flattening the list.
    """
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


def sum_dice_pools(a: tuple[tuple[int, ...], ...], b: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """
    Elementwise sum of two dice-pool-shaped tuples ((black), (blue), (red)).
    Used to merge two single-die picks (e.g. from dice_pair_choices) into one removal.
    """
    return tuple(tuple(x + y for x, y in zip(face_a, face_b)) for face_a, face_b in zip(a, b))


def dice_pair_choices(attack_pool_result: tuple[tuple[int, ...], ...]) -> list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    """
    All unordered pairs of 2 dice (by color/face) selectable together from the
    current attack pool, honoring available counts: a face can be picked twice
    only if at least 2 such dice remain in the pool.

    Each element of a returned pair is a one-hot dice-pool tuple in the same
    shape produced by dice_single_choice(pool).
    """
    canonical = dice_single_choice(FULL_DICE_POOL)
    pairs = []
    for dice_a, dice_b in itertools.combinations_with_replacement(canonical, 2):
        combined = sum_dice_pools(dice_a, dice_b)
        if all(need <= have for need_face, have_face in zip(combined, attack_pool_result) for need, have in zip(need_face, have_face)):
            pairs.append((dice_a, dice_b))
    return pairs


if __name__ == "__main__":
    # --- Example Usage ---
    # Input: 1 black die, 1 blue die, 0 red dice
    dice_pool = (3,3,3)

    
    # all_possible_outcomes = generate_all_dice_outcomes(dice_pool)

    print(f"Dice Pool: {dice_pool}")
    # print(f"Total Unique Outcomes: {len(all_possible_outcomes)}\n")

    # # Print each outcome for clarity
    # for i, outcome in enumerate(all_possible_outcomes):
    #     print(f"Outcome {i+1}: {outcome}")

    dice_roll_result = roll_dice(dice_pool)
    print(dice_icon(dice_roll_result))
    # print(f'result : {dice_icon(dice_roll_result)}')
    # for dice_choice in dice_single_choice(dice_roll_result) :
    #     print(dice_choice)



