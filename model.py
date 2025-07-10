import random
import numpy as np

# 얘네는 좀 이따 생각하고 일단 랜덤 함수로 만들어놓자고
# 게임 모델부터 완성이 되어야 뭐가 좀 될듯
def random_model(output_dimension):
    return np.random.rand(output_dimension)

def shared_encoder(state):
    # return encoded gamestate for H-DL
    pass

def ship_activation(encoded_state):
    # return which ship to activate (3 ships)
    return np.argmax(random_model(3))

def choose_attacker(encoded_state):
    # return which hull to perform attack (4 hulls)
    return np.argmax(random_model(4))

def choose_defender(encoded_stat, attacker):
    # return defending hull zone (12 hulls)
    return np.argmax(random_model(12))

def choose_speed(encoded_state):
    # return speed
    return np.argmax(random_model(5))

def choose_yaw(encoded_state, speed):
    # return yaw (-2 ~ 2)
    return np.argmax(random_model(5))
