"""Standard moveset for a square lattice"""

import numpy as np

from mrrvis.move import Move, Transformation, Collision, CollisionCase
from mrrvis.cell import Square
from typing import List
import warnings

class slide(Move):
    """slide along the surface of neighbors which form a line paralel to the module"""
    cell_type = 'Square'

    compass = ['N','S','E','W']

    def generate_collision(self, module)-> Collision:
        """collision for square slide
        
        - the default cases are for a northward move which are then rotated counterclockwise to the appropriate direction
        - each move has two possible cases where it is potentially valid, for a north move these would be where there are modules to
          the west and northwest or east and northeast, so long as one or both of these cases is true the move is valid
        """

        # define collision cases for a northward move
        case0 = CollisionCase(empty= np.array([[0,1]]), full= np.array([[1,0],[1,1]]))
        case1 = CollisionCase(empty= np.array([[0,1]]), full=np.array([[-1,0],[-1,1]]))
        
        compass = {'N':0, 'W':1, 'S':2, 'E':3} # note, rotations are counterclockwise
        num_turns = compass[self.direction]

        case0 = case0.rotate(num_turns)
        case1 = case1.rotate(num_turns)
        return Collision([case0+module, case1+module], 'or')

    def generate_transaction(self) -> List[Transformation]:
        """transaction for square slide
        
        this is a single transformation where the translation is simply a move of one in a compass direction
        """

        transformation = Transformation(
            location = self.module,
            transformation= Square.adjacent_transformations(self.config.connect_type)[self.direction],
            collision= self.generate_collision(self.module)
        )
        return [transformation]

class rotate(Move):
    cell_type = 'Square'
    # checklist = []
    compass = ['NE','SE','SW','NW']

    def generate_collision(self, module) -> Collision:
        """collision object for the rotation move
        
        - collision for a rotation move requires that exactly one of two cases is true:
          for a northeast move (the default), 
          either that the the north  and northeast neighbors are empty
          or that the east and northeast neighbors are empty
        """
        # default cases for northeast move
        case0 = CollisionCase(empty=np.array([[0,1],[1,1]]), full = np.array([[1,0]]))
        case1 = CollisionCase(empty=np.array([[1,0],[1,1]]), full = np.array([[0,1]]))
        #note, rotations are counterclockwise
        compass = {'NE':0, 'NW':1, 'SW':2, 'SE':3}
        num_turns = compass[self.direction]
        case0 = case0.rotate(num_turns)
        case1 = case1.rotate(num_turns)

        return Collision([case0+module,case1+module],'xor')

    def generate_transaction(self) -> List[Transformation]:
        """transactions for this move consist of a single translation in the given direction"""
        
        transformation = Transformation(
            location = self.module,
            transformation = Square.adjacent_transformations('vertex')[self.direction],
            collision= self.generate_collision(self.module)
        )

        return [transformation]

class slide_line(Move):
    cell_type = 'Square'
    # checklist = []
    compass = ['N','S','E','W']

    def generate_collision(self)-> Collision:
        """collision for slide line (push) move
        collision for a slide line move is always true, as each cell pushes the one in front"""

        return Collision(cases=np.array([]), eval_rule=None)
    
    def generate_transaction(self) -> List[Transformation]:
        """The transaction for a slide line move
        this consists of a list of transformations, where each cell moves to the position of the one in front, until the end of the line
        """
        transformations=[]
        
        module = self.module
        base_transformation = Square.adjacent_transformations('edge')[self.direction]

        next_occupied= True
        while next_occupied:
            
            transformations.append(Transformation(
                location=module,
                transformation= base_transformation,
                collision= self.generate_collision()
            ))
            module = module+ base_transformation

            if module not in self.config: # if the next module is None, then stop looping
                next_occupied=False
        
        return reversed(transformations) # we reverse the transformations because configuration graphs don't allow duplicates
