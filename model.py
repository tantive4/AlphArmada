import numpy as np
MASK_VALUE = -10e8

def random_model(output_dimension):
    return np.random.rand(output_dimension)

def softmax(X):
    exp_a = np.exp(X)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def shared_encoder(state):
    # return encoded gamestate for H-DL
    pass

def choose_ship_activate(encoded_state = None):
    # return which ship to activate (3 ships)
    return random_model(3)

def choose_attacker(encoded_state = None):
    # return which hull to perform attack (4 hulls)
    return random_model(4)

def choose_defender(attacker, encoded_stat = None):
    # return defending hull zone (12 hulls)
    return random_model(12)

def choose_speed(encoded_state = None):
    """
    [speed 0, speed 1, speed 2, speed 3, speed 4]
    """
    return random_model(5)

def choose_yaw(speed, currenent_joint, encoded_state = None):
    """
    [left 2, left 1, straight, right 1, right 2]
    """
    return random_model(5)

def choose_placement(course, encoded_state = None) :
    """
    [right side, left side]
    """
    return random_model(2)