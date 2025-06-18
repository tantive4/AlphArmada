import random
import numpy as np

def master_model(state):
    formatted_state = [random.random() for _ in range(len(state))]
    return formatted_state

def ship_activation(formatted_state):
    policy = [random.random() for _ in range(3)]
    policy /= np.sum(policy)
    return policy

def squadron_activation(formatted_state):
    policy = [random.random() for _ in range(10)]
    policy /= np.sum(policy)
    return policy

def set_command(formatted_state):
    policy = [random.random() for _ in range(4)]
    policy /= np.sum(policy)
    return policy

def squadron_command(formatted_state):
    policy = [random.random() for _ in range(2)]
    policy /= np.sum(policy)
    return policy

def engineer_command(formatted_state):
    policy = [random.random() for _ in range(2)]
    policy /= np.sum(policy)
    return policy