from enum import IntEnum

class AttackRange(IntEnum) :
    INVALID = -1
    CLOSE = 0
    MEDIUM = 1
    LONG = 2
    EXTREME = 3

CLOSE_RANGE = 123.3
MEDIUM_RANGE = 186.5
LONG_RANGE = 304.8