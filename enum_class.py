import json
from enum import IntEnum

with open('ship_dict.json', 'r') as f:
    SHIP_DATA: dict = json.load(f)

with open('squad_dict.json', 'r') as f:
    SQUAD_DATA: dict = json.load(f)

class Player(IntEnum):
    REBEL = 1
    EMPIRE = -1
    def __str__(self):
        return self.name
    __repr__ = __str__

class Dice(IntEnum) :
    BLACK = 0
    BLUE = 1
    RED = 2
    def __str__(self):
        return self.name
    __repr__ = __str__

class HullSection(IntEnum):
    FRONT = 0
    RIGHT = 1
    REAR = 2
    LEFT = 3
    def __str__(self):
        return self.name
    __repr__ = __str__

class SizeClass(IntEnum) :
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    HUGE = 4
    def __str__(self):
        return self.name
    __repr__ = __str__

class Command(IntEnum) :
    NAV = 0
    REPAIR = 1
    CONFIRE = 2
    SQUAD = 3
    def __str__(self):
        return self.name
    __repr__ = __str__

class AttackRange(IntEnum) :
    INVALID = -1
    CLOSE = 0
    MEDIUM = 1
    LONG = 2
    EXTREME = 3
    def __str__(self) -> str:
        return self.name
    __repr__ = __str__

class Critical(IntEnum) :
    STANDARD = 0
    def __str__(self) :
        return self.name
    __repr__ = __str__