import numpy as np
MASK_VALUE = -10e8
# 얘네는 좀 이따 생각하고 일단 랜덤 함수로 만들어놓자고
# 게임 모델부터 완성이 되어야 뭐가 좀 될듯
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
    # return speed
    return random_model(5)

def choose_yaw(speed, encoded_state = None):
    # return yaw (-2 ~ 2)
    return random_model(5)
