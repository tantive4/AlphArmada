import os
import random
import shutil
from time import time
from action_phase import Phase
from enum_class import Command
from armada import setup_game
from enum_class import *
def main():
    """Main function to set up and run the game."""
    random.seed(66)
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    
    with open('simulation_log.txt', 'w') as f:
        f.write("Game Start\n")


    
    
    # cr90b, neb_b_support, cr90a, neb_b_escort, vsd1, vsd2 = game.ships
    # xwing1, xwing2, xwing3, tie1, tie2, tie3, tie4, tie5, tie6 = game.squads
    
    game = setup_game(debuging_visual=False)
    snapshot = (0.0, 4, Phase.SQUAD_ACTIVATE, 1, 1, 0, None, 2, None, 
                ((-14198.29296875, 687.1251831054688, 8.050329208374023, 3, 0, (0, 0, 0, 0), True, False, (), (), (Command.CONFIRE,), (), 0, (None, None, None, None), 0, (), {4: (False, True, False), 5: (False, True, False), 2: (False, True, False)}), 
                 (1018.4247436523438, 786.273681640625, 0.9817476868629456, 2, 5, (2, 1, 0, 1), False, True, (Command.NAV,), (), (Command.CONFIRE, Command.NAV), (), 0, (None, None, None, None), 0, (), {4: (True, False, False), 0: (True, False, False), 1: (True, False, False)}), 
                 (1175.3292236328125, 1048.7669677734375, 0.39269912242889404, 4, 0, (0, 0, 0, 0), True, False, (), (), (), (), 1, (None, None, (5, HullSection.REAR), None), 0, (), {4: (False, True, False), 5: (False, True, False), 2: (False, True, False)}), 
                 (1515.2979736328125, 826.8258666992188, 0.39269909262657166, 2, 5, (3, 0, 2, 0), False, True, (Command.SQUAD,), (), (), (), 0, (None, None, None, None), 0, (), {4: (True, False, False), 0: (True, False, False), 1: (True, False, False)}), 
                 (860.1974487304688, 278.9951171875, 2.356194496154785, 1, 8, (2, 1, 0, 2), False, True, (Command.NAV, Command.NAV), (), (Command.CONFIRE, Command.NAV), (), 0, (None, None, None, None), 0, (), {0: (True, False, False), 2: (False, False, False), 3: (True, False, False)}), 
                 (1057.7174072265625, 258.9368896484375, 3.5342917442321777, 1, 6, (1, 3, 0, 2), False, True, (Command.CONFIRE, Command.NAV), (), (Command.REPAIR,), (), 0, (None, None, None, None), 0, (), {0: (True, False, False), 2: (True, False, False), 3: (True, False, False)})), 
                 ((5, (1200.0, 281.0), False, False, False, False, None, {}), 
                  (5, (904.7853002998788, 419.5), False, False, False, False, None, {}), 
                  (5, (1392.5, 595.2146997001212), False, False, False, False, None, {}), 
                  (3, (1247.5, 862.1762239271874), False, False, False, False, None, {}), 
                  (3, (1295.7717982555089, 567.8282149624737), False, False, False, False, None, {}), 
                  (3, (1135.9238633219832, 199.49115350768724), False, False, False, False, None, {}), 
                  (3, (648.0, 835.0), False, False, False, False, None, {}), 
                  (3, (901.8121094353373, 824.9009771415635), False, False, False, False, None, {}), 
                  (3, (925.9722985663361, 681.4338691411293), False, False, False, False, None, {})))


    game.revert_snapshot(snapshot)
    game.debuging_visual = True
    for _ in range(10):
        actions = game.get_valid_actions()
        action = actions[0]
        game.apply_action(action)





    
if __name__ == '__main__':
    main()


# zip -r _game_visual.zip game_visuals
