# cython: profile=True

from enum_class import *
from itertools import product

cdef class DefenseToken:
    def __init__(self, token_type: str) -> None :
        self.type: TokenType = TokenType[token_type.upper()]
        self.readied : bool = True
        self.discarded : bool = False
        self.accuracy : bool = False

    def __str__(self):
        return f'{"Readied" if self.readied else "Exhausted"} {self.type.name}'
    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality by comparing the attribute dictionaries of the two objects.
        This automatically includes any new attributes added to the class later.
        """
        # First, check if the objects are of the same class
        if not isinstance(other, DefenseToken):
            return NotImplemented
        
        cdef DefenseToken other_token = <DefenseToken>other

        # Compare all attributes directly
        return (self.type == other_token.type and
                self.readied == other_token.readied and
                self.discarded == other_token.discarded and
                self.accuracy == other_token.accuracy)

    def spend(self) -> None :
        if self.readied:
            self.readied = False
        else : self.discard()

    def discard(self) -> None :
        self.readied = False
        self.discarded = True

    def ready(self) -> None :
        self.readied = True

    cdef tuple get_snapshot(self) :
        """
        get a snapshot of the defense token for saving and loading purpose
        """
        return (self.readied, self.discarded, self.accuracy)

    cdef void revert_snapshot(self, tuple snapshot) :
        """
        revert the defense token to a previous snapshot
        """
        self.readied, self.discarded, self.accuracy = snapshot

TOKEN_DICT = {
    2 * index + double : DefenseToken(TokenType(index).name) for index, double in product(TokenType, (0, 1))
    # 0 : Brace,
    # 1 : Brace, 
    # 2 : Redirect
    # 3 ... etc
}