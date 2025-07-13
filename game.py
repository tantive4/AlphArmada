
from armada import *
import math



game = Armada()

cr90 = Ship(CR90A_dict, 1)
nebulon = Ship(Neb_escort_dict, 1)
victory = Ship(Victory_2_dict, -1)




game.deploy_ship(cr90,600, 175, 0, 2) # id = 0
game.deploy_ship(victory,450, 725, math.pi, 2) # 1
game.deploy_ship(nebulon, 300, 175, 0, 2) # 2

game.play()