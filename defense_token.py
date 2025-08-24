from functools import wraps
from enum import Enum
def check_discarded(func):
    """Decorator to prevent methods from running if the token is discarded."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.discarded:
            raise ValueError(f"Cannot perform '{func.__name__}' on a discarded token.")
        return func(self, *args, **kwargs)
    return wrapper

def protect_all_methods(cls):
    """
    Class decorator that applies the @check_discarded decorator to all
    public methods of the class (except __init__).
    """
    # Loop through every attribute in the class dictionary
    for attr_name, attr_value in cls.__dict__.items():
        # Check if the attribute is a regular, callable method
        if callable(attr_value) and not attr_name.startswith("__"):
            # Replace the original method with the decorated version
            setattr(cls, attr_name, check_discarded(attr_value))
    return cls

@protect_all_methods
class DefenseToken:
    def __init__(self, token_type: str, token_index : int) -> None :
        self.type: TokenType = TokenType[token_type.upper()]
        self.index = token_index
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
        
        # Remove the 'index' key before comparison
        self_dict.pop('index', None)
        other_dict.pop('index', None)
        
        # Compare the modified dictionaries
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

class TokenType(Enum):
    BRACE = 'brace'
    REDIRECT = 'redirect'
    EVADE = 'evade'
    SCATTER = 'scatter'
    CONTAIN = 'contain'
    SALVO = 'salvo'