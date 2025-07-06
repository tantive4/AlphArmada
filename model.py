import random
import numpy as np

# 얘네는 좀 이따 생각하고 일단 랜덤 함수로 만들어놓자고
# 게임 모델부터 완성이 되어야 뭐가 좀 될듯

def shared_encoder(state):
    # return encoded gamestate for H-DL
    pass

def ship_activation(encoded_state):
    # return which ship to activate (3 ships)
    pass

def choose_attacker(encoded_state):
    # return which hull to perform attack (4 hulls)
    pass

def choose_defender(encoded_stat, attacker):
    # return defending hull zone (12 hulls)
    pass

def choose_speed(encoded_state):
    # return speed
    pass

def choose_yaw(encoded_state, speed):
    # return yaw (-2 ~ 2)
    pass

def use_accuracy(state):
    # return which defense token to block
    # hand coded model
    pass

def use_defense_token(state):
    # return which defense token to use and how
    #hand coded model
    pass