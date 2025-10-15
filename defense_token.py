from enum import IntEnum
from itertools import product

class DefenseToken:
    def __init__(self, token_type: str) -> None :
        self.type: TokenType = TokenType[token_type.upper()]
        self.readied : bool = True
        self.discarded : bool = False
        self.accuracy : bool = False

    def __str__(self):
        return f'{'Readied' if self.readied else 'Exhausted'} {self.type.name}'
    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality by comparing the attribute dictionaries of the two objects.
        This automatically includes any new attributes added to the class later.
        """
        # First, check if the objects are of the same class
        if not isinstance(other, DefenseToken):
            return NotImplemented
        
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()

        return self_dict == other_dict

    def spend(self) -> None :
        if self.readied:
            self.readied = False
        else : self.discard()

    def discard(self) -> None :
        self.readied = False
        self.discarded = True

    def ready(self) -> None :
        self.readied = True

class TokenType(IntEnum):
    BRACE = 0
    REDIRECT = 1
    EVADE = 2
    SCATTER = 3

TOKEN_DICT = {
    2 * index + double : DefenseToken(TokenType(index).name) for index, double in product(TokenType, (0, 1))
    # 0 : Brace,
    # 1 : Brace, 
    # 2 : Redirect
    # 3 ... etc
}