import numpy as np


RNG = np.random.default_rng()

# 22 anonymous damage card types.  Faceup card effects are not implemented, so a
# type is identified only by its index and its copy count: the 7th type has 4
# copies, the 22nd has 8, every other type has 2.
DAMAGE_CARD_COPIES = (
    2, 2, 2, 2, 2, 2, 4,
    2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2,
    8,
)
DAMAGE_CARD_TYPES = len(DAMAGE_CARD_COPIES)
DAMAGE_DECK_SIZE = sum(DAMAGE_CARD_COPIES)

EMPTY_DAMAGE_CARDS = (0,) * DAMAGE_CARD_TYPES

# Repair payload sentinel: discard an anonymous facedown card rather than a
# specific faceup type.
FACEDOWN_CARD_INDEX = DAMAGE_CARD_TYPES

# Why the game is sitting in Phase.DRAW_DAMAGE_CARD.  The two sources deal the
# card differently: a critical only turns an already-dealt facedown card faceup,
# while an asteroid deals a brand new card.
DAMAGE_SOURCE_NONE = 0
DAMAGE_SOURCE_ATTACK = 1
DAMAGE_SOURCE_OBSTACLE = 2


def draw_damage_card(dealt_counts: tuple[int, ...]) -> int:
    """
    Samples one damage card type without replacement from a player's deck.

    Each player draws from their own 52 card deck.  Facedown cards are ignored,
    so only the faceup cards already held by that player's surviving ships are
    excluded from the deck.

    Args:
        dealt_counts: per-type count of that player's faceup cards.

    Returns:
        the drawn card type index.
    """
    remaining = []
    total = 0
    for copies, dealt in zip(DAMAGE_CARD_COPIES, dealt_counts):
        left = copies - dealt
        if left < 0:
            raise ValueError(f'More damage cards dealt than exist in the deck: {dealt_counts}')
        remaining.append(left)
        total += left

    if total <= 0:
        raise ValueError('Damage deck is empty; no discard pile is modeled to reshuffle.')

    # Cumulative walk rather than RNG.choice(p=...): this runs inside the MCTS
    # selection loop, once per traversal through a damage node.
    threshold = RNG.random() * total
    cumulative = 0
    for index, left in enumerate(remaining):
        cumulative += left
        if threshold < cumulative:
            return index
    return DAMAGE_CARD_TYPES - 1


if __name__ == '__main__':
    print(f'{DAMAGE_CARD_TYPES} types, {DAMAGE_DECK_SIZE} cards')
    counts = [0] * DAMAGE_CARD_TYPES
    for _ in range(52000):
        counts[draw_damage_card(EMPTY_DAMAGE_CARDS)] += 1
    print([round(count / 1000, 2) for count in counts])
