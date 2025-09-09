import numpy as np

class DummyModel:
    """
    A placeholder model that returns random outputs for value and policy.
    This is used for testing the MCTS structure without a trained neural network.
    """
    def __init__(self, action_manager):
        """
        Initializes the dummy model.
        
        Args:
            action_manager: An instance of the ActionManager class to get
                            the size of the action space for each phase.
        """
        self.action_manager = action_manager

    def __call__(self, state, phase):
        """
        Mimics the forward pass of the neural network.
        
        Args:
            state: The current game state (not used by the dummy model, but
                   included for API consistency with the real model).
            phase: The current GamePhase enum member.
            
        Returns:
            A tuple containing:
            - policy (np.ndarray): A randomly generated policy vector.
            - value (float): A random value between -1 and 1.
        """
        # Get the total number of possible actions for the given phase.
        action_map = self.action_manager.get_action_map(phase)
        action_size = len(action_map['total_actions'])
        
        # 1. Generate a random policy vector of the correct size.
        #    The probabilities are raw and don't sum to 1 yet.
        #    The MCTS search function will handle normalization after masking.
        policy = np.random.rand(action_size)
        
        # 2. Generate a single random value for the state evaluation.
        #    The value is drawn from a standard normal distribution.
        value = np.random.rand()
        
        return policy, value