"""A module defining the control environment of a reconfiguration problem"""

from collections import deque
from typing import NamedTuple, Union

from mrrvis.cell import Cell, Square, Tri, Cube, Hex
from mrrvis.movesets import square
from mrrvis.configuration import ConfigurationGraph
import numpy as np
from mrrvis.move import Move

cell_registry = {
    "Square": Square,
    "Cube": Cube,
    "Triangle": Tri,
    "Hexagon": Hex,
}
default_movesets = {
    Square: {
        "slide": square.slide,
        "rotate": square.rotate,
    },
    Hex: {},
    Tri: {},
    Cube:{}
}


    
class Environment():
    def __init__(self, cell_type: str, moveset:dict = None, 
        ruleset:dict = None, initial_config: np.ndarray = None, target_config: np.ndarray = None, buffer_size: int = 10000):

        if issubclass(type(cell_type), Cell): # assign the cell type
            self.Cell = cell_type
        else:
            try:
                self.Cell = cell_registry[cell_type]
            except KeyError:
                raise KeyError(f"{cell_type} is not a valid cell type")

        self.moveset = default_movesets[self.Cell] if moveset is None else moveset
        self.ruleset = {} if ruleset is None else ruleset
        self.config = (
            ConfigurationGraph(self.Cell, initial_config) if initial_config is not None 
            else ConfigurationGraph(self.Cell) # Future: this could be replaced with a random configuration generator, but for now we'll just use the default
        )

        self.target_config = ConfigurationGraph(self.Cell, target_config) if target_config is not None else None
        self.history = deque(buffer_size)

    @property
    def n(self):
        return self.config.n

    def move_compasses(self):
        """Return the action space of the environment"""
        moves = list(self.moveset.values())
        return [(move.__name__, move.compass) for move in moves]



        

    configuration: ConfigurationGraph
    target: ConfigurationGraph
    moves: dict[Move]
    configuration_rules: dict
    history: deque

class EnvironmentWrapper:
    """A wrapper for an environment which allows stepwise interaction with the environment"""
    def __init__(self, environment: Environment):
        self.environment = environment
    
    def step(self, move: str):
        """Perform a step in the environment"""
        try:
            move = self.environment.moves[move]
        except KeyError:
            raise ValueError("Move not found")
        
        
        
        return self.environment





def make(Cell_type: Cell, initial_configuration: Union[ConfigurationGraph, np.ndarray], target: ConfigurationGraph = None, moves: dict = None, configuration_rules: dict = None, hist_buffer_size = 10000) -> Environment:
    """Make an environment"""
    if isinstance(initial_configuration, ConfigurationGraph):
        configuration = initial_configuration
    else:
        configuration = ConfigurationGraph(Cell_type, initial_configuration)
    if moves is None:
        moves = {}
    if configuration_rules is None:
        configuration_rules = {}
    env = Environment(Cell_type, configuration, target, moves, configuration_rules, deque(maxlen= hist_buffer_size))
    return EnvironmentWrapper(env)



# def move(Move:'mrrvis.move.Move', module: np.array, direction: str, graph: ConfigurationGraph, check_connectivity=True)-> ConfigurationGraph:
#     """
#     Move a vertex to a new location
    
#     move will be an instance of a subclass of Move, 
#     if the move exists, then this will contain the transformation as a tuple
#     if the move is infeasible then this move object will be None
#     """
#     move = Move(module, direction, graph)
#     if move() is None:
#         return graph