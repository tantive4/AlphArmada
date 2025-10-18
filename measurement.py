import numpy as np

from enum_class import SizeClass



CLOSE_RANGE = 123.3
MEDIUM_RANGE = 186.5
LONG_RANGE = 304.8

DISTANCE = {i : distance for i, distance in enumerate((0, 77, 125, 185, 245, 304.8))} # check


SHIP_BASE_SIZE : dict[SizeClass, tuple]= {SizeClass.SMALL : (43, 71), SizeClass.MEDIUM : (63, 102), SizeClass.LARGE : (77.5, 129)}
SHIP_TOKEN_SIZE :  dict[SizeClass, tuple] = {SizeClass.SMALL : (38.5, 70.25), SizeClass.MEDIUM : (58.5, 101.5)}
TOOL_WIDTH_HALF : float= 15.25 / 2
TOOL_LENGTH : float= 46.13 
TOOL_PART_LENGTH : float = 22.27

SQUAD_BASE_RADIUS : float = 16.875 # check
SQUAD_TOKEN_RADIUS : float = 16 # check
Q2S_RANGE = DISTANCE[1] + SQUAD_TOKEN_RADIUS
Q2Q_RANGE = DISTANCE[1] + SQUAD_TOKEN_RADIUS * 2

ERROR_EPSILON = 1e-9
ROTATION_MATRICES = np.stack([
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),         # 0 deg
    np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),        # 90 deg CW
    np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),       # 180 deg CW
    np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)         # 270 deg CW
])
SQUAD_TOKEN_POLY = np.array([
    [np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, num=60, endpoint=False, dtype=np.float32)
], dtype=np.float32) * SQUAD_TOKEN_RADIUS
