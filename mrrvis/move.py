"""
The logic for moving a module within the configuration
"""

from abc import ABC, abstractmethod
import copy
from gettext import translation
from typing import List, NamedTuple, Tuple, Union
import numpy as np
from mrrvis import Cell, ConfigurationGraph, connected
from mrrvis.geometry_utils import rotate
import warnings


class CollisionCheck(NamedTuple):
    empty: np.ndarray
    full: np.ndarray

    def rotate(self, turns, base_angle=np.pi/2, around=None, axis=None)-> 'CollisionCheck':
        return CollisionCheck(
            rotate(self.empty,turns,base_angle, around, axis), 
            rotate(self.full, turns, base_angle, around, axis)
        )




class Transformation(NamedTuple):
    location: int
    translation: np.ndarray
    collisions: List[CollisionCheck]


class Checkwrapper:
    """A variation on the Failure Monad design pattern
    which allows us to pipeline checks on the feasibility of a configuration

    a useful explanation is given at
    https://medium.com/swlh/monads-in-python-e3c9592285d6
    """

    def __init__(self, value: ConfigurationGraph) -> None:
        self.value = value

    def get(self):
        return self.value

    def bind(self, f: callable) -> 'Checkwrapper':
        """Bind any function that takes a ConfigurationGraph and returns a boolean"""
        if self.get() is None:
            return Checkwrapper(None)

        with warnings.catch_warnings():
            #if the check fails, we want to ignore any warnings it may throw
            #this is because we are checking the feasibility of the configuration
            warnings.simplefilter("ignore")

            if f(self.value):
                return self
            else:
                return Checkwrapper(None)



class Move(ABC):
    Transformation = Transformation
    Collision = CollisionCheck

    def __init__(self, graph: ConfigurationGraph, module: Union[int, np.ndarray], direction, additional_checks=[], check_connectivity=True):
        """an object which generates a move transformation with context
        if the move would be invalid then the move is said to not exist.

        subclasses must implement the following:
        :param: compass: list
            a compass explaining the directions that a move can be carried out
        :method: _generate_transforms(graph, module_id, direction) -> list[Transformation,...]
            a method which generates a list of named transformation tuples 
        :method: _generate_collisions(direction, transformations)-> list[collision,...]
            a method which generates a list of named collision tuples
        """
        # check module id is valid
        if type(module) == int:
            try:
                graph[module]
            except IndexError:
                raise IndexError(f"index {module} is out of range")
        elif type(module) == np.array:
            try:
                module = graph.get_index(module)
            except UserWarning:
                raise ValueError(f"{module} is not in the graph")


        self.transformations = self.generate_transforms(
            graph, module, direction)

        if self.no_collision(graph, self.transformations):
            new_graph = self.transform(graph, self.transformations)

            self.wrapped_value = Checkwrapper(new_graph)

        else:
            self.wrapped_value = Checkwrapper(None)

        print('value: ' ,self.wrapped_value.get())
        

        #perform the checks
        if check_connectivity:
            self.wrapped_value = self.wrapped_value.bind(connected)
            print('value: ' ,self.wrapped_value.get())

        #add additional checks to the checklist
        self.checklist+= additional_checks
 
        # perform the checks on the wrapped monad, if it's value survives the checks then the move is valid
        for check in self.checklist:
            self.wrapped_value = self.wrapped_value.bind(check)


    @classmethod
    def no_collision(cls,graph: ConfigurationGraph, transformations: List[List[Transformation]], collision_rule='or') -> bool:
        """
        :param graph: a graph object
        :param collisions: a list of named collision tuples
        :return: a boolean indicating whether the transformations collide with the graph

        collision_rule can be 'and' or 'or' or 'xor'

        'or' is the default, but for rotating moves, 'xor' is usually required, because the module will have to rotate
        around something else to get to the target location
        """
        collision_rule = cls.collision_rule

        def evaluate_case(module_id:int,graph:ConfigurationGraph, case: CollisionCheck) -> bool:
            transform_location = graph.vertices[module_id]
            # print(transform_location)
            
            empty = transform_location + case.empty
            full = transform_location + case.full
            # we need to make sure that the empty and full arrays are 2D arrays by adding a dimension if their row length is 1
            if len(empty.shape) == 1:
                empty = np.array([empty])
            if len(full.shape) == 1:
                full = np.array([full])
            for vertex in empty:
                if vertex.tolist() in graph.vertices.tolist():
                    return False
            for vertex in full:
                if vertex.tolist() not in graph.vertices.tolist():
                    return False
            return True

        collision_rules = {
            'or': lambda cases: any(cases),
            'xor': lambda cases: sum(cases) == 1,
            'and': lambda cases: all(cases)
        }
        if collision_rule not in list(collision_rules.keys()):
            raise ValueError(
                f"collision rule {collision_rule} is not a valid collision rule")

        transformation_collisions = []
        for transformation in transformations:
            module_id = transformation.location
            cases = []

            [cases.append(evaluate_case(module_id, graph, case))
             for case in transformation.collisions]
            # print(cases)
            transformation_collisions.append(collision_rules[collision_rule](cases))
        print(transformation_collisions)
        return all(transformation_collisions)

    
    @staticmethod
    def transform(graph: ConfigurationGraph, transformations: List[Transformation]) -> ConfigurationGraph:
        """

        :param graph: a graph object
        :param transformations: a list of named transformation tuples
        :return: a new graph object with the transformations applied
        """
    
        new_graph = copy.deepcopy(graph)
        for transformation in transformations:
            new_graph.vertices[transformation.location] += transformation.translation
        return new_graph

    def __call__(self) -> Union[ConfigurationGraph, None]:
        return self.wrapped_value.get()

    @classmethod
    @property
    @abstractmethod
    def cell_type(cls) -> Cell:
        """the type of the cell that the move is designed for"""
        pass

    @classmethod
    @property
    @abstractmethod
    def checklist(cls) -> List[callable]:
        """a list of checks to be performed on the target configuration, specific to that move"""
        pass

    @classmethod
    @property
    @abstractmethod
    def collision_rule(cls) -> str:
        """The collision rule for the move, typically sliding moves use 'or', while rotation moves use 'xor'"""
        pass

    @classmethod
    @abstractmethod
    def generate_transforms(cls, graph, module_id, direction) -> List[Transformation]:
        pass

    @classmethod
    @abstractmethod
    def generate_collisions(cls, direction, collision_index=0) -> Tuple[CollisionCheck, ...]:
        pass