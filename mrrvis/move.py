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
from typing import Callable, Iterable, List, Literal, NamedTuple, Union

import numpy as np
from mrrvis.configuration import ConfigurationGraph, edge_connected, vert_connected
from mrrvis.geometry_utils import rotate_normal
import warnings


class CollisionCase(NamedTuple):
    """A single collision case"""
    empty: np.ndarray
    full: np.ndarray

    def rotate(self, turns, base_angle=np.pi/2, around:np.ndarray=None, axis=None) -> 'CollisionCase':
        """rotate both arrays in the collision object
        :param turns: the number of times the base angle to rotate by (counter clockwise)
        :param base_angle: the angle per turn
        :param around: the coordinate to rotate around
        :param axis: the axis or normal vector to rotate around
        :type axis: str or np.ndarray
        This comes in handy when we need to rotate a basic collision example around a compass for a move
        """
        return CollisionCase(
            rotate_normal(self.empty, turns, base_angle, around, axis),
            rotate_normal(self.full, turns, base_angle, around, axis)
        )

    def evaluate_case(self, graph: ConfigurationGraph, ) -> bool:
        """evaluate collision for this case
        :param graph: graph to check collision against
        :return: a bool showing true if no collision detected
        """

        empty = self.empty
        full = self.full
       
        if len(empty.shape) == 1: # we need to make sure that the empty and full arrays are 2D arrays
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

    def __call__(self) -> bool:
        return self.evaluate_case()

    def __add__(self, other: np.ndarray):
        return CollisionCase(self.empty+other, self.full+other)


class Collision(NamedTuple):
    """a collection of collision cases which have some evaluation rule when called
    :param cases: the collection of cases to evaluate
    :param eval_rule: the rule governing the evaluation of this collision"""
    cases: List[CollisionCase]
    eval_rule: Literal['or','and','xor']

    def evaluate_feasible(self, graph)-> bool:
        """evaluation of collision
        :param graph: the graph to evaluate against
        :return: true if no collision, false if one is detected
        """

        # if the collision object contains no cases we assume it is true
        if len(self.cases)==0:
            return True

        collision_rules = {
            'or': lambda cases: any(cases),
            'xor': lambda cases: sum(cases) == 1,
            'and': lambda cases: all(cases)
        }
        case_evaluations = [case.evaluate_case(graph) for case in self.cases]
        try:
            return collision_rules[self.eval_rule](case_evaluations)
        except KeyError as e:
            warnings.warn('collision rule must be one of or|xor|and')
            raise e

    def __call__(self, graph):
        return self.evaluate_feasible(graph)


class Transformation(NamedTuple):
    """The transformation and collision information of a single module in a transaction

    :param location: the module that is being affected
    :param transformation: the resulting location of the module(s) (can include replication)
    :param collision: the Collision object for this transformation
    """
    location: np.ndarray
    transformation: np.ndarray
    collision: Collision


class Checkwrapper:
    def __init__(self, value: ConfigurationGraph) -> None:
        """wrap a configuration graph as a checkwrapper
        The CheckWrapper object allows for checks on the validity of a configuration to be pipelined, 
        such that if one check fails and the move in fact has no value, we can skip through the rest of the pipeline without issue
        
        :param value: the graph to be wrapped
        attributes:
            value: the graph within the wrapper
        methods:
            get: return the value
            __call__: returns self.get
            bind: bind a function to the value, where if the function returns true, the value is unchanged, but if false
                self.value becomes None as the configuration is infeasible
    
        A variation on the Failure Monad design pattern
        which allows us to pipeline checks on the feasibility of a configuration

        a useful explanation is given at
        https://medium.com/swlh/monads-in-python-e3c9592285d6
        """
        self.value = value

    def get(self):
        "get the unwrapped value"
        return self.value

    def bind(self, f: callable) -> 'Checkwrapper':
        """Bind any function that takes a ConfigurationGraph and returns a boolean
        :param f: the function to bind
        :return: the checkwrapper after the bound function

        if true, then return the current wrapped value of the graph
        if false, then return wrapped None
        if a previous check has failed and the value is already None, then return None again
        """
        if self.value is None:
            return Checkwrapper(None)

        try:
            if f(self.value) is True:
                
                return Checkwrapper(self.value)
            else:

                return Checkwrapper(None)

        except Exception as e:
            raise TypeError(f"failed with exception \n{e}. \n\
            Functions passed into the checkwrapper have to take a graph as input and return a bool") from e
    def __call__(self):
        return self.get()


class Move(ABC):

    @classmethod
    @property
    @abstractmethod
    def compass(cls) -> List[str]:
        """a list of valid compass directions for this move"""
        pass

    @classmethod
    @property
    @abstractmethod
    def cell_type(cls) -> Literal['Square','Hex','Tri','Cube']:
        """the type of the cell that the move is designed for"""
        pass




    def __init__(self, configuration: ConfigurationGraph, module: Union[int, np.ndarray], direction:str, additional_checks:List[Callable]=None, verbose=False, check_connectivity=True):
        """A callable object which can be used to safely move objects within a configuration, subject to rules set by the environment
        
        arguments:
        :param configuration: The configuration graph to be edited
        :param module: the module to be moved
        :param direction: the direction for the move to be carried out in (see self.__class__.compass to get the compass of this move)
        :param additional_checks: environment specific checks that the resulting configuration would have to take, on top of those already defined by the move.
            such should be passed as a list of functions where those functions take a Configuration graph as input and return a bool as output
        :param verbose: whether to provide warning information to explain the infeasability of a move

        (methods and attributes marked S need to be implemented in the concrete subclass)
        methods:
            S generate_collisions: builds a Collision object for a single transformation
            S generate_transaction: builds a transaction attribute which is a list of Task objects which govern the local feasibility of the transformation
            attempt_transaction: takes the attribute
            evaluate_checklist: evaluate global feasibility of the resulting configuration using the checklist
        attributes:
            S compass: the valid directions for the move
            S cell_type: the type of cell that the move is valid for
            S checklist: the list of checks to be carried out upon the candidate configuration
        """

            
        if type(module) == int: #get the module location in the graph if the index was given
            try:
                module = configuration[module]
            except KeyError as e:
                warnings.warn("the module must be either an index in the graph, or a coordinate in its vertices")
                raise e
        module= np.array(module)


        if not (direction in self.compass): #check string literal
            raise ValueError(f"direction must be in {self.compass}")

        self.checklist = []
        if check_connectivity: #add the basic graph connectivity function to the checklist
            connect_funcs = {
                'vertex': vert_connected,
                'edge': edge_connected,
                'face': None
            }

            self.checklist.append(connect_funcs[configuration.connect_type]) 

        if additional_checks is not None:
            self.checklist = self.checklist + additional_checks #append externally required checks to the checklist

        self.config = configuration
        self.module = module
        self.direction = direction.upper()
        self.verbose = verbose

    @abstractmethod
    def generate_collision(self, module)-> Collision:
        """Generate the collision for a single transformation
        must be implemented in subclass
        remark. Collisions are objects containing a list of CollisionCase Objects and a collision rule, which affects the evaluation
        """
        pass
    @abstractmethod
    def generate_transaction(self)-> Iterable[Transformation]:
        """generate a list of Transformation objects which need to be completed
        must be implemented in subclass

        recall, a task contains a module 
        """
        pass


    def attempt_transaction(self, transaction: Iterable[Transformation]) -> Union[ConfigurationGraph, None]:
        """attempt to construct a new graph from the transaction
        :param transaction: the list of transformations to be performed in this move
        :return: the graph after this transaction is applied or none
        """
        if transaction is None:
            # This indicates that the transaction is invalid
            return None

        new_graph = copy.deepcopy(self.config)
        for transformation in transaction: 

            if not transformation.collision(self.config): #test collision
                warnings.warn('Collision: move infeasible')
                return None
            if len(transformation.transformation.shape) != 1:
                raise NotImplementedError("transformation for replication not implemented yet")


            index = new_graph.get_index(transformation.location)
            new_graph.vertices[index] += transformation.transformation

        return new_graph


    def evaluate_checklist(self, candidate: ConfigurationGraph):
        """evaluate the checklist on the resulting configuration
        :param candidate: the graph to be evaluated
        :return: candidate if feasible or None, if the move is infeasible
        """
        wrapped_value = Checkwrapper(candidate)

        for check in self.checklist:
            wrapped_value = wrapped_value.bind(check)

        return wrapped_value.get()
        

    def evaluate(self)-> Union[ConfigurationGraph, None]:
        """evaluate the validity of a move

        :return: graph after having completed the move, or None if the move is infeasible

        to do this we need to complete three steps:
        1. generate a transaction (implemented by subclasses)
        2. attempt transaction, to assert the local feasibility of the move
        3. evaluate the checklist on the resulting configuration
        """
        if self.module not in self.config:  
            warnings.warn('module not in graph')
            return None
        #1 generate a transaction
        transaction = self.generate_transaction()
        #2 attempt transaction
        candidate_graph = self.attempt_transaction(transaction)
        #3 evaluate new configuration against the ruleset
        value = self.evaluate_checklist(candidate_graph)

        return value

    def __call__(self)-> Union[ConfigurationGraph, None]:
        """generates the transaction, attempts it and then evaluates the checklist pipeline, returning a value if the move is feasible"""
        return self.evaluate()


        
        

