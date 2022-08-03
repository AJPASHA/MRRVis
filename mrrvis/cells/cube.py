from  .cell import Cell
import numpy as np
"""Cube Lattice Cell"""
class Cube(Cell):
    dimensions = 3
    n_parameters = 3
    def __init__(self, coord: np.array) -> None:
        super().__init__(coord)
        #for cubic cells .neighbors we are gonna have to have 3 levels of strictness, corner, edge and facet connectivity
    

    
    def adjacents(self, connectivity: str = 'edge'):
        """for cubic lattices there are three levels of strictness so we define them as strings
        connectivity = ['face'|'edge'|'vertex']
        """
        pass

    def __dir__(self):
        pass
