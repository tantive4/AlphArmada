from armada import Armada
from ship import Ship
import math
import json
import os, shutil




if os.path.exists("game_visuals"): shutil.rmtree("game_visuals")

with open('ship_info.json', 'r') as f:
    ship_data : dict[str, dict[str, str | int | list]]= json.load(f)
with open('simulation_log.txt', 'w') as f:
    f.write("Game Start\n")


game = Armada()

cr90 = Ship(ship_data['CR90_Corvette'], 1)
nebulon = Ship(ship_data['NebulonB_Escort'], 1)
victory = Ship(ship_data['Victory_SD'], -1)


game.deploy_ship(cr90,600, 175, 0, 2) # id = 0
game.deploy_ship(victory,450, 725, math.pi, 2) # 1
game.deploy_ship(nebulon, 300, 175, 0, 2) # 2

player1 = lambda: game.mcts_decision(iterations=400)
player2 = lambda: game.mcts_decision(iterations=200)
# player2 = game.random_decision
game.play(player1, player2)

# zip -r game_visuals.zip game_visuals

# NEED TO FIX DEFENSE TOKEN MECHANISM!!!