"""The history module defines the History object, which stores a sequence of configurations"""
from mrrvis.configuration import ConfigurationGraph
from collections import deque

class History:
    """A History of Configurations
    
    Parameters
    ----------
    state_0: ConfigurationGraph
        The initial configuration
    
    Attributes
    ----------
    history: collections.deque
        A queue of the configuration history
    cell_type: str
        The type of cell used by this object's configurations

    Methods
    -------
    __iter__()
        Iterate through history 
    __reversed__()
        obtain the reverse of the history
    __contains__(other)    

    """
    def __init__(self, state_0: ConfigurationGraph):
        self.history = deque() # this might need some buffer size setting in the future
        self.history.append(state_0)
        self.cell_type = state_0.Cell.__name__
    @property
    def t(self):
        """The amount of items (time steps) in the history"""
        return len(self.history)

    def append(self, state: ConfigurationGraph):
        """Append a move to the history
        
        Parameters
        ----------
        state: ConfigurationGraph
            the state to add
        """

        self.history.append(state)


    def __getitem__(self, item):
        return self.history[item]

    def revert(self, k: int = 1):
        """revert the history by n steps

        Parameters
        ---------- 
        k:  int
            the number of steps to revert by
        
        Returns
        -------
        ConfigurationGraph
            The last state after the reversion
        """
        for _ in range(k):
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