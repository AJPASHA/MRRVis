import numpy as np

from mrrvis.move import Move, Transformation, Collision, CollisionCase

from mrrvis.cell import Hex
from typing import List, Iterable
import warnings

class rotate(Move):
    """A rotation of 60 degrees on a hexagonal configuration"""
    cell_type = 'Hex'

    compass = ['N','NE','SE','S', 'SW','NW']

    def generate_collision(self, module: np.ndarray) -> Collision:
        """Collision for hexagonal rotation move
        
        remarks. 
        - collision for hex rotation is an xor collision case where one of the neighbors in the rotation direction must be open and the other must be closed
        - we use the northbound case as a base, which we then rotate around the origin and add to the specific module location
        """
        # default cases for north movement
        case0 = CollisionCase(empty=np.array([[0,1,-1]]), full = np.array([[-1,1,0]]))
        case1 = CollisionCase(empty=np.array([[0,1,-1]]), full = np.array([[1,0,-1]]))
        compass = {
            'N':0,
            'NW':1,
            'SW':2,
            'S':3,
            'SE':4,
            'NE':5
        }
        num_turns = compass[self.direction]

        case0 = case0.rotate(num_turns,np.pi/3)
        case1 = case1.rotate(num_turns,np.pi/3)
        return Collision([case0+module, case1+module], 'xor')


    def generate_transaction(self) -> Iterable[Transformation]:
        """Transaction for hexagonal rotation move
        
        for this we only need one transformation as this is a single component move
        """
                
        transformation = Transformation(
            location = self.module,
            transformation = Hex.adjacent_transformations('vertex')[self.direction],
            collision= self.generate_collision(self.module)
        )

        return [transformation]