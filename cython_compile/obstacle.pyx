import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos


from armada cimport Armada
from enum_class import ObstacleType
from measurement import STATION, DEBRIS1, DEBRIS2, ASTEROID1, ASTEROID2, ASTEROID3

cdef class Obstacle:
    def __init__(self, object type, int copy=1):
        self.type = type
        if type == ObstacleType.STATION :
            self.coordinates = STATION.copy()
            self.index = 0

        elif type == ObstacleType.DEBRIS :
            if copy == 1 :   self.coordinates = DEBRIS1.copy() ; self.index = 1
            else :           self.coordinates = DEBRIS2.copy() ; self.index = 2

        elif type == ObstacleType.ASTEROID :
            if copy == 1 :   self.coordinates = ASTEROID1.copy() ; self.index = 3
            elif copy == 2 : self.coordinates = ASTEROID2.copy() ; self.index = 4
            else :           self.coordinates = ASTEROID3.copy() ; self.index = 5

    def __str__(self):
        return self.type.name
    __repr__ = __str__

    cpdef void place_obstacle(self, float x, float y, float orientation, bint flip):
        self.x = x
        self.y = y
        self.orientation = orientation
        rotation_matrix = np.array([[cos(orientation), -sin(orientation)],
                                    [sin(orientation),  cos(orientation)]])
        
        # 1. Create a (2, 1) column vector for translation
        translation = np.array([x, y], dtype=np.float32).reshape((2, 1))

        # 2. Transpose coordinates from (N, 2) to (2, N) for multiplication
        #    We assume 'self.coordinates' holds the BASE shape of the obstacle
        coords_T = self.coordinates.T 

        if flip:
            flip_matrix = np.array([[-1, 0],
                                    [ 0, 1]])
            
            # Apply transformations: (2x2) @ (2x2) @ (2xN) + (2x1)
            transformed_coords_T = rotation_matrix @ (flip_matrix @ coords_T) + translation
        else:
            # Apply transformations: (2x2) @ (2xN) + (2x1)
            transformed_coords_T = rotation_matrix @ coords_T + translation

        # 3. Transpose the final result back from (2, N) to (N, 2)
        #    This overwrites self.coordinates with the new, transformed position.
        self.coordinates = transformed_coords_T.T

        self.hash_state = (self.index, self.x, self.y, self.orientation, flip)

    cpdef tuple get_hash_state(self):
        return self.hash_state