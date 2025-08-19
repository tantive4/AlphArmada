# To add type annotations, we need to import the 'Callable' type from the 'typing' module.
# A 'Callable' is something that can be called, like a function or a method.
from typing import Callable
import random

# It can be helpful to create a type alias for complex types to make the code cleaner.
# Here, we're defining 'DecisionMethod' as a callable that takes no arguments (besides 'self')
# and returns a string representing the move.
# The format is Callable[[list_of_arg_types], return_type].
# An empty list [] means the method takes no arguments.
DecisionMethod = Callable[[], str]

class Game:
    """
    A simple demonstration class for a game where different decision-making
    methods can be used by players.
    """
    def __init__(self, starting_player: int = 1):
        """Initializes the game state."""
        self.current_player = starting_player
        self.turn = 1
        print(f"--- New Game Started ---")

    # --- Decision Methods ---
    # These are the methods that can be passed as arguments. Note that they all
    # have the same "signature": they take 'self' as an argument and return a string.

    def random_decision(self) -> str:
        """A simple strategy that returns a random move."""
        moves = ["rock", "paper", "scissors"]
        return f"Randomly chose '{random.choice(moves)}'"

    def greedy_decision(self) -> str:
        """A placeholder for a more complex 'greedy' strategy."""
        # In a real game, this would look at the board and make the best immediate move.
        return "Greedily chose 'paper' to beat 'rock'"

    def mcts_decision(self) -> str:
        """A placeholder for a Monte Carlo Tree Search strategy."""
        # This is a sophisticated AI technique for games like Go or Chess.
        return "Used MCTS to determine the best move is 'scissors'"

    # --- The Main Game Logic Method ---

    def play(self, player1_strategy: DecisionMethod, player2_strategy: DecisionMethod, num_turns: int = 4):
        """
        Simulates playing the game for a number of turns.

        Args:
            player1_strategy: The method player 1 will use to make decisions.
            player2_strategy: The method player 2 will use to make decisions.
            num_turns: The number of turns to simulate.
        """
        print(f"Player 1 is using: {player1_strategy.__name__}")
        print(f"Player 2 is using: {player2_strategy.__name__}\n")

        while self.turn <= num_turns:
            print(f"Turn {self.turn}:")
            move_description = ""
            if self.current_player == 1:
                # Here, we call the method that was passed in as an argument.
                # player1_strategy is a variable that holds a reference to a method
                # like self.random_decision. Calling player1_strategy() is the
                # same as calling self.random_decision().
                move_description = player1_strategy()
                print(f"  Player 1: {move_description}")
                self.current_player = 2 # Switch to the next player
            else:
                # Call player 2's chosen strategy.
                move_description = player2_strategy()
                print(f"  Player 2: {move_description}")
                self.current_player = 1 # Switch back to player 1

            self.turn += 1
            print("-" * 20)


# --- How to use it ---
if __name__ == "__main__":
    # Create an instance of the Game class.
    my_game = Game()

    # Now, call the play method, passing the actual methods as arguments.
    # Notice we don't use parentheses () after the method names here.
    # We are passing the method object itself, not the result of calling it.
    my_game.play(
        player1_strategy=my_game.random_decision,
        player2_strategy=my_game.greedy_decision
    )

    # You can easily play another game with different strategies.
    another_game = Game()
    another_game.play(
        player1_strategy=my_game.greedy_decision,
        player2_strategy=my_game.mcts_decision
    )
