"""contains the Move class and its dependancies

Move is a callable object which produces a new graph configuration if possible or returns None if not.

CollisionCheck is a NamedTuple which contains two fields:
    - empty: the locations which need to be empty for a move to be feasible
    - full: the locations which need to be full for a move to be feasible
CollisionCheck represents a single collision case for a move, one move can contain multiple collision cases

Transformation is a NamedTuple which contains three fields:
    - location: the index of the module to be moved
    - translation: the translation vector of the move
    - collisions: a list of CollisionCheck objects
Transformation represents a single module transformation, one move can contain multiple transformations if more than one module is relocated

Checkwrapper is a way to wrap a candidate configuration so that additional checks can be made upon it for testing the validity
    - of the resulting move
    - it simply takes a ConfigurationGraph as a value, wraps it and allows checks to be made on its' feasibility
    - these checks are simply functions which take a graph as their only argument and return a boolean

"""

from abc import ABC, abstractmethod
import copy
from typing import Callable, Iterable, List, Literal, NamedTuple, Union

import numpy as np
from mrrvis.configuration import ConfigurationGraph, edge_connected, vert_connected
from mrrvis.geometry_utils import rotate_normal
import warnings


class CollisionCase(NamedTuple):
    """A collision case
    Parameters
    ----------
    empty: ndarray
        the array of coordinates which need to be empty for the case to be true
    full: ndarray
        the array of coordinates which need to be full for the case to be true
    
    """
    empty: np.ndarray
    full: np.ndarray

    def rotate(self, turns, base_angle=np.pi/2, around:np.ndarray=None, axis=None) -> 'CollisionCase':
        """rotate both arrays in the collision object
        
        Parameters
        ----------
        turns: int
            the number of times the base angle to rotate by (counter clockwise)
        base_angle: float
            the angle to rotate per turn
        around: ndarray, optional
            the coordinate to rotate around
        axis: str 
            the axis string (some combination of ['x','y','z']) or normal vector to rotate around

        Notes
        -----
        This comes in handy when we need to rotate a basic collision example around a compass for a move
        """
        return CollisionCase(
            rotate_normal(self.empty, turns, base_angle, around, axis),
            rotate_normal(self.full, turns, base_angle, around, axis)
        )

    def evaluate_case(self, graph: ConfigurationGraph, ) -> bool:
        """evaluate collision for this case

        Parameters
        ----------
        graph: Configuration Graph
            graph to check collision against
        
        Returns
        --------
        bool
            true if no collision detected
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
    """a collection of collision cases which can be evaluated with a number of different behaviours
    
    Parameters
    ----------
    cases: list
        the collection of cases to evaluate
    eval_rule: {'or','and', 'xor'}
        the rule governing the evaluation of this collision 

    Notes
    -----
    A Collision is feasible if the list of collision cases meets the given condition:
            'or' means that the collision is feasible if any case is true
            'xor' means that the collision is feasible if exactly one case is true
            'and' means that the collision is feasible if all cases are True
        
    """
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

    Parameters
    ----------
    location: ndarray
        the coordinates of the module that is being transformed
    transformation: ndarray
        the resulting location of the module(s) (can include replication by inputting a 2D array here)
    collision: Collision
        The Collision object to evaluate for this transformation
    """
    location: np.ndarray
    transformation: np.ndarray
    collision: Collision


class Checkwrapper:
    """Wrap a ConfigurationGraph in this object to pipeline checks on the feasibility of that configuration
    
    Parameters
    ----------
    value: ConfigurationGraph
        The graph to be wrapped
    
    Attributes
    ----------
    value: ConfigurationGraph
        The graph within the wrapper
    
    Notes
    -----
    CheckWrapper is a variation on the Failure Monad design pattern which allows us to pipeline checks on the feasibility of a configuration.

    a useful explanation is given at
    https://medium.com/swlh/monads-in-python-e3c9592285d6
    """
    def __init__(self, value: ConfigurationGraph) -> None:
        self.value = value

    def unwrap(self):
        "get the unwrapped value"
        return self.value

    def bind(self, f: callable) -> 'Checkwrapper':
        """Bind any function that takes a ConfigurationGraph and returns a boolean to the wrapped value

        Parameters
        ----------
        f: func
            the function to bind

        Returns
        -------
        CheckWrapper
            A new CheckWrapper which contains a value if the move still appears feasible

        Notes
        -----
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
        return self.unwrap()


class Move(ABC):
    """A Move which can be called as a function to generate a new configuration, if one would exist for the move so defined
    
    Parameters
    ----------
    configuration: ConfigurationGraph
        The configuration graph to be edited
    module: int or ndarray
        The module to be moved
    direction: str 
        The direction for the move to be carried out in (see self.__class__.compass to get the compass of this move)
    additional_checks: list of funcs
        environment specific checks that the resulting configuration would have to take, on top of those already defined by the move.
        such should be passed as a list of functions where those functions take a Configuration graph as input and return a bool as output
    verbose: bool 
        whether to provide warning information to explain the infeasability of a move

    Attributes
    ----------
    checklist: list of funcs 
        the list of checks to be carried out upon the candidate configuration

    Methods
    -------
    __call__()
        evaluate the move's feasibility and return the resulting configuration, if one exists

    """

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
    def generate_collision(self, module:np.ndarray)-> Collision:
        """Generate the collision for a single transformation

        must be implemented in subclass
        
        Parameters
        ----------
        module: ndarray
            The coordinates of the module to generate the collision for

        Returns
        -------
        Collision
            A collision to evaluate

        Notes
        -----
        must be implemented in subclass
        
        Collisions are objects containing a list of CollisionCase Objects and a collision rule, which affects the evaluation

        """
        pass
    @abstractmethod
    def generate_transaction(self)-> Iterable[Transformation]:
        """generate a list of Transformation objects which need to be completed
        

        Returns
        -------
        list of Transformation
            The list of Transformations to carry out 

        Notes
        -----
        must be implemented in subclass
        """
        pass


    def attempt_transaction(self, transaction: Iterable[Transformation]) -> Union[ConfigurationGraph, None]:
        """attempt to construct a new graph from the transaction
        
        Parameters
        ----------
        transaction: list of Transformations
            the list of transformations to be performed in this move
        
        Returns
        -------
        ConfigurationGraph or None
            the graph after this transaction is applied or none
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

        Parameters
        ----------
        candidate: ConfigurationGraph the graph to be evaluated
        
        Returns
        -------
        ConfigurationGraph or None
            candidate if feasible or None, if the move is infeasible
        """
        wrapped_value = Checkwrapper(candidate)

        for check in self.checklist:
            wrapped_value = wrapped_value.bind(check)

        return wrapped_value.unwrap()
        

    def evaluate(self)-> Union[ConfigurationGraph, None]:
        """evaluate the validity of a move

        :return: graph after having completed the move, or None if the move is infeasible

        Notes
        -----
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


        
        

