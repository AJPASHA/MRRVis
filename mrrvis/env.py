"""A module defining the control environment of a reconfiguration problem"""

from collections import deque
from typing import Callable, Literal, NamedTuple, Tuple, Union, Dict
import warnings
from matplotlib.animation import FuncAnimation
from mrrvis.cell import Cell, Square, Tri, Cube, Hex
from mrrvis.vistools import plot_configuration, plot_history, default_style
from mrrvis.movesets import squaremoves, hexmoves
from mrrvis.history import History
from mrrvis.configuration import ConfigurationGraph
import numpy as np
from mrrvis.move import Move

cell_registry = {
    "Square": Square,
    "Cube": Cube,
    "Tri": Tri,
    "Hex": Hex,
}
default_movesets = {
    'Square': {
        "slide": squaremoves.slide,
        "rotate": squaremoves.rotate,
        "push": squaremoves.slide_line
    },
    'Hex': {
        'rotate': hexmoves.rotate
    },
    'Tri': {},
    'Cube': {}
}


class Environment:
    def __init__(self, state_0: Union[np.ndarray, ConfigurationGraph], cell_type: Literal['Square', 'Cube', 'Tri', 'Hex'], state_target: Union[np.ndarray, ConfigurationGraph] = None, moveset: dict = None, connectivity: Literal['edge', 'vertex', 'face'] = None,
        additional_rules = None):
        """A class representing the control environment of a reconfiguration problem as a Markov Process
        Parameters:
        :param state_0: a set of vertices representing the initial configuration
        :param state_target: a set of vertices representing the target configuration
        :param cell_type: the type of cell to use for the configuration graph. Must be one of
                ["Square", "Triangle", "Cube", "Hexagon"]
        :param moveset: a dictionary of movesets for the cell type
        :param connectivity: 
        Attributes:
            state: np.ndarray. The current state of the environment
            target_state: ConfigurationGraph. The target state of the environment
            moveset: dictionary. a dictionary of movesets for the cell type
            history: History. an object containing the transformation history of the environment
            action_space: dict. The dictionary structure containing moves for all modules and all moves.
        Methods:
            step(move_name, module, direction): (S_t+1, R_t+1, done) perform a move in place on env, and return the next state, a simple reward function, and a 
                bool stating whether the new configuration matches a target state
            verify(S_t+1): Bool. Verify if the next state is the target state
            reward(S_t+1): int. apply the reward function to the next state
            render(): None. render the current state of the environment
            renderhistory(): None. Generate an animation of the environment
            reset(random=False): restore the environment to its original configuration or some random configuration
            revert(steps=1): restore the environment to its state, a number of steps ago

        """
        cell_type = cell_type.capitalize() # capitalise the cell type, just to avoid annoying input errors

        if connectivity is None:
            if cell_type in ['Square', 'Tri', 'Cube']:
                connectivity = 'vertex'
            if cell_type == 'Hex':
                connectivity = 'edge'

        self.connectivity = connectivity



        self.state = ConfigurationGraph(cell_type, np.array(state_0), connect_type=self.connectivity) if type(
            state_0) != ConfigurationGraph else state_0

        if state_target is None:
            self.target_state = None
        else:
            self.target_state = ConfigurationGraph(cell_type, np.array(state_target), connect_type=self.connectivity) if type(
                state_target) != ConfigurationGraph else state_target

        self.moveset = moveset if moveset is not None else default_movesets[cell_type]
        self.history = History(self.state)
        
        if additional_rules is None:
            self.ruleset = []
        else:
            self.ruleset = list(additional_rules)


    @property
    def t(self):
        """returns the number of time steps in this environment"""
        return self.history.t

    def step(self, move_name:str, module:Union[np.ndarray, int], direction:str) -> Tuple[ConfigurationGraph, int, bool]:
        """take a step in the environment, returning the new state, reward, and a flag to indicate if the environment has reached the target state
        
        params:
        :param move_name: the name of the move, which must be a key in this environment's moveset
        :param module: either the id or coordinate location of a module in this environment's state
        :param direction: a string signifying a direction in the compass of the chosen move
        returns:
        :return: the configuration of the next state
        :return: reward for this move, by default -1 on all non-terminal moves
        :return: whether the next state matches the environment target state (is a terminal state)
        """

        action = self.moveset[move_name](
            self.state, module, direction, check_connectivity=True, additional_checks=self.ruleset)
        if action() is None:
            warnings.warn("infeasible action selected")
            return self.state, 0, False

        self.state = action()
        self.history.append(self.state)
        return self.state, self.reward(self.state), self.verify(self.state)

    def auto_step(self, next_state: ConfigurationGraph)-> Tuple[ConfigurationGraph, int, bool]:
        """step from an already specified action (useful for automation)
        params:
        :param: next_state: the configuration of the next state
        returns:
        :return: reward: reward for this move, by default -1 on all non-terminal moves
        :return: done: whether the next state matches the environment target state (is a terminal state)

        """
        if next_state is None:
            warnings.warn("infeasible action selected")
            return 0, False

        self.state = next_state
        self.history.append(next_state)
        return self.reward(self.state), self.verify(self.state)

    # To generate the action space we need to generate the action space for each module, move combination, then combine them
    # This allows us to reference a move in the action space as env.action_space[module_name][move_name][direction]

    @property
    def action_space(self) -> 'Dict[str, Dict[str, Move]]':
        """get the current action space

        :return: a dictionary of the action space

        Preferably, if you know the module and the move name, 
        just use env.actions_for_module_and_move(module, move_name)[direction] 
        or, if you know the module, use env.actions_for_module(module)[move_name][direction]
        to save time and memory
        """
        vertices = self.state.vertices
        return {i: self.actions_for_module(vertices[i]) for i in range(len(vertices))}

    def actions_for_module(self, module: np.ndarray) -> 'Dict[str, Dict[str, Move]]':
        """return the actions for a particular module (actions_for_module_and_move over all moves)

        :param module: the coordinate of a vertex in the configuration
        :return: the dictionary of the actions for that module
        """
        return {move_name: self.actions_for_module_and_move(module, move_name) for move_name in self.moveset.keys()}

    def actions_for_module_and_move(self, module: np.ndarray, move_name: str) -> 'Dict[str, Move]':
        """return the actions for a specified module and move
        params:
        :param module: the coordinate of a vertex in the configuration
        :param move_name: the name of a move from this environment's moveset
        
        """
        try:
            move: Move = self.moveset[move_name]
        except KeyError:
            raise ValueError('move name should be in the current moveset')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return {direction: move(self.state, module, direction)() for direction in self.moveset[move_name].compass}

    def add_move(self, move: Move, alias: str = None):
        """add a move to the moveset
        params:
        :param move: is a move to add to the moveset
        :param alias: the key to give this move in the moveset dictionary
        """
        if alias is None:
            alias = move.__name__
        self.moveset[alias] = move
    def remove_move(self, move_name: str):
        """remove a move from the moveset with the given key
        params:
        :param move_name: the key of the move to remove
            
        """
        try:
            del self.moveset[move_name]
        except KeyError:
            raise KeyError("Move name must be in the current moveset keys, which can be accessed by Environment.moveset.keys()")

    def verify(self, state_next: ConfigurationGraph = None) -> bool:
        """verify that a state matches the target state
        params:
        :param state_next: the state to compare against the target, leave blank to use this env's current state
        :return: bool stating if the input state is isomorphic to the target state
        """
        if self.target_state is None:
            return None
        if state_next is None:
            return self.target_state == self.state
        return self.target_state == state_next

    def reward(self, state_next: ConfigurationGraph) -> int:
        """simple reward function, for automation.
        gives -1 for all non-terminal states
        """
        if self.target_state is None:
            return None
        return 0 if self.verify(state_next) else -1

    def reset(self) -> ConfigurationGraph:
        """reset the environment to its original configuration"""

        self.state = self.history[0]
        self.history = History(self.state)  # reset the history

        return self.state

    def revert(self, k:int=1) -> ConfigurationGraph:
        """revert the environment to its state k steps ago, returning the new state

        :param steps: the number of steps to revert by
 
        :return: the configuration graph k steps ago
        """
        self.history.revert(k)
        self.state = self.history[-1]
        return self.state

    def render(self, axes=False,show=True, save=False, filename=None, **style):
        """render the current state of the environment
        :param axes: whether to show axes
        :param show: whether to show graph
        :param save: whether to save graph
        :param filename: what name to save the graph under any matplotlib kwargs can be added using the **style attribute
        """

        plot_configuration(self.state, axes=axes,show=show, save=save, filepath=filename, **style)

    def render_history(self, speed: int = 200, show=True, save=False, filename=None, **style) -> FuncAnimation:
        """render the history of this environment as an animation, can be saved as a gif or mp4
        :param speed: the length of each frame in milliseconds
        :param show: whether to show graph
        :param save: whether to save graph
        :param filename: what name to save the graph under
        any matplotlib kwargs can be added using **style
        """

        return plot_history(self.history, speed=speed, show=show, save=save, filepath=filename, **style)
