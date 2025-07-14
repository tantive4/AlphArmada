
from armada import Armada
from ship import Ship
import math
import json




game = Armada()

with open('ship_info.json', 'r') as f:
    ship_data = json.load(f)
    
cr90 = Ship(ship_data['CR90_Corvette'], 1)
nebulon = Ship(ship_data['NebulonB_Escort'], 1)
victory = Ship(ship_data['Victory_SD'], -1)




game.deploy_ship(cr90,600, 175, 0, 2) # id = 0
game.deploy_ship(victory,450, 725, math.pi, 2) # 1
game.deploy_ship(nebulon, 300, 175, 0, 2) # 2

game.play()