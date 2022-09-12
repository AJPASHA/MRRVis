"""defines a history object for storing a series of configurations"""
from mrrvis.configuration import ConfigurationGraph
from mrrvis.move import Move
from collections import deque

class History:
    def __init__(self, state_0: ConfigurationGraph):
        """The history of an environment
        

        :param state_0: the initial state of the environment system
        attributes:
        :param history: a queue of the states of the system
        :param cell_type: the type of cell which the graph uses
        """
        self.history = deque() # this might need some buffer size setting in the future
        self.history.append(state_0)
        self.cell_type = state_0.Cell.__name__
    @property
    def t(self):
        """The amount of items (time steps) in the history"""
        return len(self.history)
    def append(self, state: ConfigurationGraph):
        """Append a move to the history

        :param state: the state to add
        """

        self.history.append(state)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, item):
        return self.history[item]

    def revert(self, n: int = 1):
        """revert the history by n steps
        params: 
        :param n: the number of steps to revert by
        returns:
        :return: The last state after the reversion
        """
        for _ in range(n):
            self.history.pop()
        return self.history[-1]

    def __iter__(self):
        return iter(self.history)

    def __reversed__(self):
        return reversed(self.history)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"""History, t={self.t}\ncell type = {self.cell_type}\ncurrent state: {self[-1].vertices} """

    def __hash__(self):
        return hash(self.history)

    def __contains__(self, item: ConfigurationGraph):
        return item in self.history