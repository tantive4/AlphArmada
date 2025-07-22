
from armada import Armada
from ship import Ship, HullSection
import math
import json
from mcts import MCTS, MCTSState
import copy


game = Armada()

with open('ship_info.json', 'r') as f:
    ship_data : dict[str, dict[str, str | int | list]]= json.load(f)
    
cr90 = Ship(ship_data['CR90_Corvette'], 1)
nebulon = Ship(ship_data['NebulonB_Escort'], 1)
victory = Ship(ship_data['Victory_SD'], -1)


game.deploy_ship(cr90,600, 175, 0, 2) # id = 0
game.deploy_ship(victory,450, 725, math.pi, 2) # 1
game.deploy_ship(nebulon, 300, 175, 0, 2) # 2

active_ship = cr90
active_ship_id = active_ship.ship_id

state : MCTSState= {
    "game": game,
    "current_player": game.current_player,
    "decision_phase": "attack", # The entry point for the attack macro-action
    "active_ship_id": active_ship_id,
    "maneuver_speed": None, "maneuver_course": [], "maneuver_joint_index": 0,
    "attack_hull": None, "defend_ship_id": None, "defend_hull": None
}
mcts_state_copy = copy.deepcopy(state)
mcts_state_copy['game'].simulation_mode = True
mcts = MCTS(initial_state=mcts_state_copy, player=game.current_player)



game.play()

# zip -r game_visualizer.zip game_visualizer