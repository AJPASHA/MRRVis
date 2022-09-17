"""The env module defines the Environment object which acts as the control environment of an mrrvis system

The Environment allows us to perform the following key operations:
- Initialize an environment with a history and a target state
"""

from typing import Literal,  Tuple, Union, Dict
import warnings

from mrrvis.cell import Square, Tri, Cube, Hex
from mrrvis.vistools import plot_configuration, plot_history
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
    """The control environment of a Reconfigurable Robotics system
    
    Parameters
    ----------
    state_0: ConfigurationGraph or ndarray
        The initial state of the configuration
    cell_type: {'Square', 'Cube', 'Tri', 'Hex'}
        The type of cell lattice used
    state_target: ConfigurationGraph or ndarray, optional
        The target configuration
    moveset: dict, optional
        A dictionary of moves, indexed by human readable names, defaults for the cell_type will be used if left blank
    connectivity: {'edge', 'vertex', 'face'}
        The level of connection at which two cells are considered neighbors, defaults to most generous afforded by cell type
    additional_rules: list of func
        Any additional configuration rules to be applied to all moves in this environment. These should be functions which take
        a ConfigurationGraph as input and return a boolean truth value

    Attributes
    ----------
    state: ConfigurationGraph
        The current state of the environment
    target_state: ConfigurationGraph
        The target state of the environment
    moveset: dict of Move
        A dictionary containing the moves cells can carry out and their names for reference
    history: mrrvis.history.History
        A history containing the history of states since the environment was initialised/reset
    """
    def __init__(self, state_0: Union[np.ndarray, ConfigurationGraph], cell_type: Literal['Square', 'Cube', 'Tri', 'Hex'], state_target: Union[np.ndarray, ConfigurationGraph] = None, moveset: dict = None, connectivity: Literal['edge', 'vertex', 'face'] = None,
        additional_rules = None):
        cell_type = cell_type.capitalize() # capitalise the cell type, just to avoid annoying input errors

        if connectivity is None:
            if cell_type == 'Cube':
                connectivity = 'face'
            if cell_type in ['Square', 'Tri']:
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
        """request to take a step in the environment, performing move in place
        
        Parameters
        ----------
        move_name: str
            the name of the move
        module: ndarray or int
            The identity of the module to move
        direction: str
            A direction in this move's compass
        
        Warns
        -----
        UserWarning
            If the move is infeasible, will cause state not to update
            
        Returns
        -------
        ConfigurationGraph
            the configuration of the next state
        int
            reward for this move, by default -1 on all non-terminal moves
        bool
            whether the next state matches the environment target state (is a terminal state)
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
        """request to make a step with a move (force step)

        Parameters
        ----------        
        next_state: 
            the configuration of the next state
        
        Returns
        -------
        int
            reward for this move, by default -1 on all non-terminal moves
        bool
            flag to say if the state matches the target state
        """


        self.state = next_state
        self.history.append(next_state)
        return self.reward(self.state), self.verify(self.state)

    # To generate the action space we need to generate the action space for each module, move combination, then combine them
    # This allows us to reference a move in the action space as env.action_space[module_name][move_name][direction]

    @property
    def action_space(self) -> 'Dict[str, Dict[str, Move]]':
        """get the current action space

        Notes
        -----
        Preferably, if you know the module and the move name, 
        just use env.actions_for_module_and_move(module, move_name)[direction] 
        or, if you know the module, use env.actions_for_module(module)[move_name][direction]
        to save time and memory
        """
        vertices = self.state.vertices
        return {i: self.actions_for_module(vertices[i]) for i in range(len(vertices))}

    def actions_for_module(self, module: np.ndarray) -> 'Dict[str, Dict[str, Move]]':
        """return the actions for a particular module

        Parameters
        ----------
        module: ndarray or int
            the coordinate or index of a vertex in the configuration
        Returns
        -------
        dict
            The dictionary of the actions for that module

        See Also
        --------
        actions_for_module_and_move: actions_for_module_and_move
        """
        return {move_name: self.actions_for_module_and_move(module, move_name) for move_name in self.moveset.keys()}

    def actions_for_module_and_move(self, module: np.ndarray, move_name: str) -> 'Dict[str, Move]':
        """return the actions for a specified module and move

        Parameters
        ----------
        module: ndarray or int
            the coordinate or index of a vertex in the configuration
        Returns
        -------
        dict
            The dictionary of the actions for the specified move and module

        See Also
        --------
        actions_for_module: actions_for_module
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

        Parameters
        ----------
        move: Move
            The move to add
        alias: str, optional
            The name of the move, default is to use name of the move
        """
        if alias is None:
            alias = move.__name__
        self.moveset[alias] = move

    def remove_move(self, move_name: str):
        """remove a move from the moveset with the given key

        Parameters
        ----------
        move_name: str
            name of the move to remove
        
        Raises
        ------
        KeyError
            If the move_name is not in the moveset
            
        """
        try:
            del self.moveset[move_name]
        except KeyError:
            raise KeyError("Move name must be in the current moveset keys, which can be accessed by Environment.moveset.keys()")

    def verify(self, state_next: ConfigurationGraph = None) -> bool:
        """verify that a state matches the target state
        
        Parameters
        ----------
        state_next: ConfigurationGraph, optional
            the state to compare against the target, leave blank to use this env's current state
        
        Returns
        -------
        bool 
            True if and only if the input state is isomorphic to this environment's target state
        """
        if self.target_state is None:
            warnings.warn('The target state is not set for this environment, please set before calling Environment.verify')
            return None
        if state_next is None:
            return self.target_state == self.state
        return self.target_state == state_next

    def reward(self, state_next: ConfigurationGraph) -> int:
        """simple reward function, for automation.
        gives -1 for all non-terminal states

        Parameters
        ----------
        state_next: ConfigurationGraph
            The configuration to reward
        
        Returns
        -------
        int
            The reward provided to the state
        """
        if self.target_state is None:
            return None
        return 0 if self.verify(state_next) else -1

    def reset(self) -> ConfigurationGraph:
        """reset the environment to its original configuration.
        
        Returns
        -------
        ConfigurationGraph
            The first state in this environment's history
        """

        self.state = self.history[0]
        self.history = History(self.state)  # reset the history

        return self.state

    def revert(self, k:int=1) -> ConfigurationGraph:
        """revert the environment to its state k steps ago, returning the new state
        
        Parameters
        ----------
        k: int
            the number of steps to revert by

        Returns
        -------
        the configuration graph k steps ago
        """
        self.history.revert(k)
        self.state = self.history[-1]
        return self.state

    def render(self, axes=False,show=True, save=False, filename=None, **style):
        """render the current state of the environment

        Parameters
        ----------
        axes: bool
            whether to show axes
        show: bool
            whether to show graph
        save: bool
            whether to save graph
        filename: str
            what name to save the graph under 
        style: dict
            see matplotlib.pyplot reference for valid kwargs
        """

        plot_configuration(self.state, axes=axes,show=show, save=save, filepath=filename, **style)

    def render_history(self, speed: int = 200, show=True, save=False, filename=None, **style) -> str:
        """render the current state of the environment

        Parameters
        ----------
        speed: int
            The length of each frame in milliseconds
        show: bool
            whether to show graph
        save: bool
            whether to save graph
        filename: str
            what name to save the graph under 
        style: dict
            see matplotlib.pyplot reference for valid kwargs
        
        Returns
        -------
        str
            A HTML representation of the video
        """

        return plot_history(self.history, speed=speed, show=show, save=save, filepath=filename, **style)
