"""Standard moveset for a square lattice"""

import numpy as np
from mrrvis.configuration import ConfigurationGraph
from mrrvis.move import CollisionCheck, Move, Transformation
from mrrvis.cells import Square
import mrrvis.cell as cell
from typing import List, Tuple, Union

class slide(Move):
    """
    Move a module within the configuration
    """
    cell_type = Square
    checklist = []
    collision_rule = 'or'

    def __init__(self, graph: ConfigurationGraph, module_id: Union[int, np.array], direction: str, checks=[]):
        super().__init__(graph, module_id, direction, checks)
    
    @classmethod
    def generate_transforms(cls, graph: ConfigurationGraph, module_id: Union[int, np.array], direction)->List[Transformation]:
        """ 
        Generate a list of transformations for a move
        :param graph: the graph to move within
        :param module_id: the id of the module to move
        :param direction: the direction to move the module in
        :return: a list of transformations
        """

        Cell = graph.Cell

        if direction not in ['N', 'S', 'E', 'W']:
            raise ValueError('Invalid direction')
        
    
        module = graph[module_id]
        # generate the transformation object
        # collisions = cls.generate_collisions(direction)
        transformation = cls.Transformation(
            location=module_id, 
            translation = Cell(module).adjacent_transformations('edge')[direction],
            collisions = cls.generate_collisions(direction)
            )
        return [transformation]

    @classmethod
    def generate_collisions(cls, direction)->Tuple[CollisionCheck,...]:
        """
        Generate a list of collisions for a move
        :param direction: the direction to move the module in
        :param transformations: the list of transformations to check for collisions
        :return: a list of collision objects
        """
        #by default, cases are for a north move

        case0 = cls.Collision(empty=np.array([0,1]), full = np.array([[1,0],[1,1]]))
        case1 = cls.Collision(empty=np.array([0,1]), full = np.array([[-1,0],[-1,1]]))
        #note, rotations are counterclockwise
        compass = {'N':0, 'W':1, 'S':2, 'E':3}
        try:
            num_turns = compass[direction]
        except KeyError:
            raise ValueError(f"{direction} is not a valid compass direction for this move")

        case0 = case0.rotate(num_turns)
        case1 = case1.rotate(num_turns)

        return (case0,case1)

class rotate(Move):
    cell_type = Square
    checklist = []
    collision_rule = 'xor'

    def __init__(self, graph: ConfigurationGraph, module: Union[int, np.ndarray], direction, additional_checks=[], check_connectivity=True):
        super().__init__(graph, module, direction, additional_checks, check_connectivity)

    @classmethod
    def generate_transforms(cls, graph: ConfigurationGraph, module_id: Union[int, np.array], direction)->List[Transformation]:
        """generate transformations for the rotation move

        """
        if direction not in ['NE', 'SE', 'SW', 'NW']:
            raise ValueError('Invalid direction')


        Cell = graph.Cell
        module = graph[module_id]
        # generate the transformation object
        transformation = cls.Transformation(
            location=module_id, 
            translation = Cell(module).adjacent_transformations('vertex')[direction],
            collisions = cls.generate_collisions(direction)
        )
        return [transformation]

    @classmethod
    def generate_collisions(cls, direction)->Tuple[CollisionCheck,...]:
        """generate collisions for the rotation move
        """
        #by default, cases are for a north east move
        case0 = cls.Collision(empty=np.array([0,1],[1,1]), full = np.array([[1,0]]))
        case1 = cls.Collision(empty=np.array([1,0],[1,1]), full = np.array([[0,1]]))
        #note, rotations are counterclockwise
        compass = {'NE':0, 'NW':1, 'SW':2, 'SE':3}
        try:
            num_turns = compass[direction]
        except KeyError:
            raise ValueError(f"{direction} is not a valid compass direction for this move")

        case0 = case0.rotate(num_turns)
        case1 = case1.rotate(num_turns)

        return (case0,case1)

    

        
