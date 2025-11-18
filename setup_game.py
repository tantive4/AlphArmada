import math

from armada import Armada
from ship import Ship
from squad import Squad
from obstacle import Obstacle
from enum_class import *


def setup_game(*,debuging_visual:bool=False, para_index:int=0) -> Armada: 

    game = Armada(initiative=Player.REBEL, para_index=para_index) 
    game.debuging_visual = debuging_visual
    rebel_ships = (
        Ship(SHIP_DATA['CR90B'], Player.REBEL),
        Ship(SHIP_DATA['Neb-B Support'], Player.REBEL),  
        Ship(SHIP_DATA['CR90A'], Player.REBEL),
        Ship(SHIP_DATA['Neb-B Escort'], Player.REBEL))


    rebel_squads = (Squad(SQUAD_DATA['X-Wing'], Player.REBEL) for _ in range(3))

    empire_ships = (
        Ship(SHIP_DATA['VSD1'], Player.EMPIRE),
        Ship(SHIP_DATA['VSD2'], Player.EMPIRE),)

    empire_squads = (Squad(SQUAD_DATA['TIE Fighter'], Player.EMPIRE) for _ in range(6))

    rebel_ship_deployment :list[tuple[float, float, float]] = [(600, 175, math.pi/16), (700, 175, math.pi/16), (1200, 175, 0), (1400, 175, 0)]
    empire_ship_deployment :list[tuple[float, float, float]] = [(600, 725, math.pi*7/8), (1200, 725, math.pi)]

    for i, ship in enumerate(rebel_ships) :
        game.deploy_ship(ship, *rebel_ship_deployment[i], 3)
    for i, ship in enumerate(empire_ships): 
        game.deploy_ship(ship, *empire_ship_deployment[i], 2)

    for i, squad in enumerate(rebel_squads) :
        game.deploy_squad(squad,  1200 + i * 50, 250)
    for i, squad in enumerate(empire_squads) :
        game.deploy_squad(squad,  1000 - i * 50, 650)

    obstacles = (
        Obstacle(ObstacleType.ASTEROID, 1),
        Obstacle(ObstacleType.ASTEROID, 2),
        Obstacle(ObstacleType.ASTEROID, 3),
        Obstacle(ObstacleType.DEBRIS, 1),
        Obstacle(ObstacleType.DEBRIS, 2),
        Obstacle(ObstacleType.STATION, 1),
    )
    obstacle_placement :list[tuple[float, float, float, bool]] = [
        (600, 400, math.pi/4, False),
        (900, 600, math.pi/2, True),
        (1300, 500, math.pi*3/4, False),
        (800, 300, math.pi/6, False),
        (1100, 400, math.pi/3, True),
        (1500, 300, 0.0, False),
    ]
    for obstacle, (x, y, orientation, flip) in zip(obstacles, obstacle_placement):
        game.place_obstacle(obstacle, x, y, orientation, flip)

    return game