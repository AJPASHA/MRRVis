"""contains the Move class and its dependencies

Move is a callable object which produces a new graph configuration if possible or returns None if not.

CollisionCheck is a NamedTuple which contains two fields:
    empty: the locations which need to be empty for a move to be feasible
    full: the locations which need to be full for a move to be feasible
CollisionCheck represents a single collision case for a move, one move can contain multiple collision cases

Transformation is a NamedTuple which contains three fields:
    location: the index of the module to be moved
    translation: the translation vector of the move
    collisions: a list of CollisionCheck objects
Transformation represents a single module transformation, one move can contain multiple transformations if more than 
    one module is relocated

Checkwrapper is a way to wrap a candidate configuration so that additional checks can be made upon it for testing the validity
    of the resulting move
    it simply takes a ConfigurationGraph as a value, wraps it and allows checks to be made on its' feasibility
    these checks are simply functions which take a graph as their only argument and return a boolean


"""


from abc import ABC, abstractmethod
import copy
from gettext import translation
from typing import List, NamedTuple, Tuple, Union
import numpy as np
from mrrvis.cell import Cell
from mrrvis.configuration import ConfigurationGraph, connected
from mrrvis.geometry_utils import rotate_normal
import warnings
from inspect import signature

class CollisionCheck(NamedTuple):
    empty: np.ndarray
    full: np.ndarray

    def rotate(self, turns, base_angle=np.pi/2, around=None, axis=None) -> 'CollisionCheck':
        return CollisionCheck(
            rotate_normal(self.empty, turns, base_angle, around, axis),
            rotate_normal(self.full, turns, base_angle, around, axis)
        )

    def evaluate_case(self, module_id: int, graph: ConfigurationGraph, ) -> bool:
        transform_location = graph.vertices[module_id]

        empty = transform_location + self.empty
        full = transform_location + self.full
        # we need to make sure that the empty and full arrays are 2D arrays by adding a dimension if their row length is 1
        if len(empty.shape) == 1:
            empty = np.array([empty])
        if len(full.shape) == 1:
            full = np.array([full])
        for vertex in empty:
            if vertex in graph:
                return False
        for vertex in full:
            if vertex not in graph:
                return False
        return True


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
            # if the check fails, we want to ignore any warnings it may throw
            # this is because we are checking the feasibility of the configuration
            warnings.simplefilter("ignore")
            try:
                if f(self.get()):
                    return Checkwrapper(self.get())
                else:
                    return Checkwrapper(None)
            except Exception as e:
                raise TypeError(f"{f.__name__} failed with exception \n{e}. \n\
                Functions passed into the checkwrapper have to take a graph as input and return a bool") from e


class Move(ABC):
    Transformation = Transformation
    Collision = CollisionCheck
    additional_collision_generators = []

    def __init__(self, graph: ConfigurationGraph, module: Union[int, np.ndarray], direction, additional_checks=[], check_connectivity=True):
        """a callable object which generates a move transformation with context

        if the move is feasible, then the transformed graph is returned, if not then None is returned on call
        if the move would be invalid then the move is said to not exist.
        Collision detection is performed on initialisation, as is connectivity checking, though the latter can be disabled

        additional checks on the feasibility of the move can be added by passing a list of functions to the additional_checks argument


        :param graph: the graph to move the module within
        :param module: the module to move; can be an index or the location coordinate of the module
        :param direction: the direction to move the module in, the compass depends on the mov

        subclasses must implement the following:
        :param(cls): compass: list
            a compass explaining the directions that a move can be carried out
        :param(cls): cell_type: Cell
            the cell type to use for the graph
        :param(cls): collision_rule: str
            the collision rule to use for the graph can be 'or', 'xor' or 'and'.
        :method: _generate_transforms(graph, module_id, direction) -> list[Transformation,...]
            a method which generates a list of named transformation tuples 
        :method: _generate_collisions(direction, transformations)-> list[collision,...]
            a method which generates a list of named collision tuples

        additionally, if the subclass has need of some global collision rule, for instance if a single transformation
        concerns multiple modules, then it can override
        :method: additional_collisions(graph, module_id, direction) -> list[CollisionCheck,...]

        'or' collision rule:
            if any of the collisions are true then the move is valid : generally useful for sliding moves
        'xor' collision rule:
            if exactly one of the collisions are true then the move is valid : generally useful for rotating moves
        'and' collision rule:
            if all of the collisions are true then the move is valid : generally useful for moves with complex pathways
        """
        # if module is an int, then we interpret it as the index of the module in the graph
        if type(module) == int:
            try:
                graph[module]
            except IndexError:
                raise IndexError(f"index {module} is out of range")

        else:   # if module is an iterable, then we interpret it as a vertex of the graph and get its index
            try:
                # this allows us to convert non-array iterables to arrays, e.g. list or set
                if hasattr(module, '__iter__'):
                    module = np.array(module)
                else:
                    raise TypeError(
                        "module must be an iterable which can be interpreted as a vector, must hasattr(module,'__iter__'")

                module = graph.get_index(module)

            except UserWarning:
                raise ValueError(f"{module} is not in the graph")

        self.transformations = self.generate_transforms(
            graph, module, direction)  # generate the list of transformations (implemented in subclass)

        additional_collisions = self.additional_collisions(graph, module, direction)
        # check if the move is locally feasible
        if self.no_collision(graph, self.transformations, additional_collisions):
            new_graph = self.transform(graph, self.transformations)

            # wrap the graph in a checkwrapper
            self.wrapped_value = Checkwrapper(new_graph)

        else:
            # the move is infeasible so wrap None instead
            self.wrapped_value = Checkwrapper(None)

        if check_connectivity:  # In general we need to check that the graph is connected after the move
            self.wrapped_value = self.wrapped_value.bind(connected)

        # add additional checks to the checklist, typically when move is called from an Environment
        self.checklist += additional_checks

        for check in self.checklist:            
            # perform the checks on the wrapped value, if it 'survives' the checks then the move is valid
            self.wrapped_value = self.wrapped_value.bind(check)

    @classmethod
    def no_collision(cls, graph: ConfigurationGraph, transformations: List[List[Transformation]], additional_collisions: list=None) -> bool:
        """
        :param graph: a graph object
        :param collisions: a list of named collision tuples
        :return: a boolean indicating whether the transformations collide with the graph

        collision_rule can be 'and' or 'or' or 'xor'

        'or' is the default, but for rotating moves, 'xor' is usually required, because the module will have to rotate
        around something else to get to the target location
        """
        collision_rule = cls.collision_rule

        collision_rules = {
            'or': lambda cases: any(cases),
            'xor': lambda cases: sum(cases) == 1,
            'and': lambda cases: all(cases)
        }
        if collision_rule not in list(collision_rules.keys()):
            raise ValueError(
                f"collision rule {collision_rule} is not a valid collision rule")

        rule = collision_rules[collision_rule]

        transformation_collisions = []
        for transformation in transformations:
            module_id = transformation.location
            cases = []

            _ = [cases.append(case.evaluate_case(module_id, graph))
             for case in transformation.collisions]
            
            _ = [cases.append(case.evaluate_case(module_id, graph)) 
                for case in additional_collisions]

            transformation_collisions.append(
                rule(cases))
            
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

    def get(self) -> Union[ConfigurationGraph, None]:
        return self.wrapped_value.get()

    @classmethod
    @property
    @abstractmethod
    def compass(cls) -> List[str]:
        """a list of compass directions"""
        pass

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
        if direction not in cls.compass:
            raise ValueError(
                f"{direction} is not a valid direction for the move")

    @classmethod
    @abstractmethod
    def generate_collisions(cls, direction, collision_index=0) -> Tuple[CollisionCheck, ...]:
        pass

    @classmethod
    def additional_collisions(cls, graph, module_id, direction) -> List[CollisionCheck]:
        return []