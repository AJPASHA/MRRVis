"""
The logic for moving a module within the configuration
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from mrrvis.graph import ModuleGraph


class Move(ABC):
    """
    Abstract class for moving a module within the configuration
    """
    
    def __init__(self, module_id: Union[int,np.array], direction:str, module_graph:ModuleGraph):
        self.graph = module_graph
        self.Cell = module_graph.Cell
        self.agent= module_id
        self.path = self.rotate_path(direction)

    @classmethod
    @property
    @abstractmethod
    def compass(self):
        """A move needs a list of compass directions, arranged in clockwise order from the north"""
        pass

    @classmethod
    @property
    @abstractmethod
    def base_path(self):
        """A move needs a list of single space translations as compass directions or compass indices
        mapping the collision boundary for a move in the 0th direction
        """
        pass

    @classmethod
    def rotate_path(cls, direction, ret_cardinal = True):
        """rotate the translation path 
        
        :param compass: list of cardinal directions in order
        :param base_path: list of compass directions to translate
        :param direction: direction of the translation
        :param ret_cardinal: return the resulting list of cardinal directions instead of the indices
        :return: list of cardinal directions or compass indices

        """
        direction = cls.compass.index(direction)

        if type(base_path[0]) == str:
            base_path = [cls.compass.index(i) for i in base_path]

        path = [(trans+direction )% len(cls.compass) for trans in base_path]

        if ret_cardinal:
            return [cls.compass[i] for i in path]
        else:
            return path

    @property
    def translation(self):
        cell_translations = self.Cell.adjacent_transformations(self.graph.connect_type)






# class Move:
#     def __init__(self, graph: ModuleGraph, vertex : Union[np.array, int], path: np.array):
#         """if the move is valid then this object will have a value which will be the same as a 
#         built in checks:
#         1. check that the module exists
#         2. check path collision
#         3. check that new graph would satisfy connectivity
#         """
#         self.value = (self
#             .bind(check_vertex)
#             .bind(check_path)
#             .bind(check_connectivity)
#             )


        


#         self.path = path
#         gt = graph

#         self.value = Gt_1


#     @classmethod
#     def unit(cls, value):
#         return cls(value)


#     def bind(self, func):
#         """returns value or None if the value is not bound"""
#         if self.value is None:
#             return None
#         else:
#             return func(self)

#     def __call__(self) -> Any:
#         return self.unit()

    

# def check_vertex(Move: Move):
#     """
#     Check if a vertex is in the graph or return None if not
#     """
#     graph = Move.graph
#     vertex = Move.vertex
#     try:
#         #not sure if this works?
#         if type(vertex) == int:
#             graph.V[vertex]
#         elif type(vertex) == np.array:
#             graph.V[graph.get_index(vertex)]

#         return Move

#     except IndexError:
#         return None

# def check_path(Move: Move):
#     """
#     Check if the path is valid or return None if not
#     """
#     graph = Move.graph
#     vertex = Move.vertex
#     path = Move.path
#     if np.any(np.all(graph.V==vertex, axis=1)):
#         return Move
#     else:
#         return None

