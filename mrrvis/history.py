from mrrvis.configuration import ConfigurationGraph
from mrrvis.move import Move
from collections import deque

class History:
    """A class to store the history of a configuration"""

    def __init__(self, state_0: ConfigurationGraph):
        self.history = deque() # this might need some buffer size setting in the future
        self.history.append(state_0)

    def append(self, move: Move):
        """Append a move to the history"""
        self.history.append(move())

    def __len__(self):
        return len(self.history)

    def __getitem__(self, item):
        return self.history[item]

    def revert(self, n: int = 1):
        """revert the history by n steps"""
        for _ in range(n):
            self.history.pop()
        return self.history[-1]

    def __iter__(self):
        return iter(self.history)

    def __reversed__(self):
        return reversed(self.history)

    def __str__(self):
        return str(self.history)

    def __repr__(self):
        return repr(self.history)

    def __hash__(self):
        return hash(self.history)

    def __contains__(self, item):
        return item in self.history