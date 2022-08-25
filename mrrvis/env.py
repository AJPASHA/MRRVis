"""A module defining the control environment of a reconfiguration problem"""

from collections import deque
from typing import  NamedTuple, Tuple, Union, Dict
import warnings
from mrrvis.cell import Cell, Square, Tri, Cube, Hex
from mrrvis.vistools import plot_discrete
from mrrvis.movesets import squaremoves
from mrrvis.history import History
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
    'Square': {
        "slide": squaremoves.slide,
        "rotate": squaremoves.rotate,
    },
    'Hex': {},
    'Tri': {},
    'Cube': {}
}




class Environment:
    def __init__(self, state_0: np.ndarray, state_target: np.ndarray, cell_type: str, moveset: dict = None):
        """A class representing the control environment of a reconfiguration problem as a Markov Process
        Parameters:
        :param: state_0: a set of vertices representing the initial configuration
        :param: state_target: a set of vertices representing the target configuration
        :param: cell_type: the type of cell to use for the configuration graph must be one of
            ["Square", "Triangle", "Cube", "Hexagon"]
        :param: moveset: a dictionary of movesets for the cell type
        Attributes:
        :attr: state: np.ndarray. The current state of the environment
        :attr: target_state: ConfigurationGraph. The target state of the environment
        :attr: moveset: dictionary. a dictionary of movesets for the cell type
        :attr: history: History. an object containing the transformation history of the environment
        :attr: action_space: dict. The dictionary structure containing moves for all modules and all moves.
        Methods:
        :meth: step(action): (S_t+1, R_t+1, terminate)
        :meth: verify(S_t+1): Bool. Verify if the next state is the target state
        :meth: reward(S_t+1): int. apply the reward function to the next state
        :meth: render(): None. render the current state of the environment
        :meth: renderhistory(): None. Generate an animation of the environment
        :meth: reset(random=False): restore the environment to its original configuration or some random configuration
        :meth: revert(steps=1): restore the environment to its state, a number of steps ago

        """
        self.state = ConfigurationGraph(cell_type, state_0) if type(state_0) == np.ndarray else state_0
        self.target_state = ConfigurationGraph(cell_type, state_target) if type(state_target) == np.ndarray else state_target
        self.moveset = moveset if moveset else default_movesets[cell_type] # does this check None?
        self.history = History(state_0)
        # self.metadata = {}

    
    def step(self, move_name, module, direction) -> Tuple[ConfigurationGraph, int, bool]:
        """take a step in the environment, returning the new state, reward, and a flag to indicate if the environment has reached the target state"""

        action = self.moveset[move_name](self.state, module, direction)
        if action() is None:
            warnings.warn("infeasible action selected")
            return self.state, 0, False
        
        self.state = action()
        self.history.append(action)
        return self.state, self.reward(self.state), self.verify(self.state)
    
    # To generate the action space we need to generate the action space for each module, move combination, then combine them
    # This allows us to reference a move in the action space as env.action_space[module_name][move_name][direction]
    @property
    def action_space(self) ->'Dict[str, Dict[str, Move]]':
        """return the current action space of the environment as a dictionary of moves

        alternatively, if you know the module and the move name, 
        just use env.actions_for_module_and_move(module, move_name)[direction] 
        or, if you know the module, use env.actions_for_module(module)[move_name][direction]
        to save time and memory
        """
        return {module: self.actions_for_module(module) for module in self.state.vertices}
    
    def actions_for_module(self, module: np.ndarray) -> 'Dict[str, Dict]':
        """return the actions for a particular module (actions_for_module_and_move over all moves)"""
        return {move_name: self.actions_for_module_and_move(module, move_name) for move_name in self.moveset.keys()}

    def actions_for_module_and_move(self, module:np.ndarray, move_name:str) -> 'Dict[str, Move]':
        """return the actions for a specified module and move"""
        return {direction: Move(module, direction) for direction in self.moveset[move_name].compass}
    

    def verify(
        self, state_next: ConfigurationGraph) -> bool: 
        return self.target_state == state_next

    def reward(self, state_next: ConfigurationGraph) -> int:
        return 0 if self.verify(state_next) else -1

    def reset(self, random=False) -> ConfigurationGraph:
        """reset the environment to its original configuration"""
        
        if not random:
            self.state = self.history[0] # set the state to the first state in the history
            self.history = History(self.state) # reset the history
        else:
            raise NotImplementedError("random reset not implemented yet")
            # self.state = self.state.random_configuration()
        
        return self.state

    def revert(self, steps=1) -> ConfigurationGraph:
        """revert the environment to its state a number of steps ago, returning the new state"""
        self.history.revert(steps)
        self.state = self.history[-1]
        return self.state

    def render(self):
        """render the current state of the environment"""
        plot_discrete(self.state)


