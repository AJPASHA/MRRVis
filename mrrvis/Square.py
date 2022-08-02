"""CellPrototype children for common lattice types
-square lattice
-cubic lattice
-hex lattice
-tri lattice
"""
from .Cell import Cell
import numpy as np

"""Square Lattice Cell"""
class Square(Cell):
    dimensions = 2
    n_parameters= 2

    def __init__(self,coord: np.array) -> None:
        """A square cell which can be considered as the prototype of the module in the module graph
        i.e. this informs the rest of the program what the shape of the cell is and how to calculate its neighbors
        """
        super().__init__(coord)

    

    def adjacents(self,connectivity='vertex') -> dict:
        """returns a dictionary of neighbors for this cell
        connectivity: if true, only returns facet (edge) connected neighbors, 
        otherwise will give both facet and edge connected neighbors
        """
        #define the set of neighborhood relations as a dictionary for the cardinal directions
        strict_base_adjacents = {'N':np.array([0,1]),'S':np.array([0,-1]),'E':np.array([1,0]),'W':np.array([-1,0])}
        weak_base_adjacents = {'NE':np.array([1,1]),'NW':np.array([-1,1]),'SE':np.array([1,-1]),'SW':np.array([-1,-1])}

        if connectivity=='edge':
            base_adjacents = strict_base_adjacents
        elif connectivity =='vertex':
            base_adjacents = {**strict_base_adjacents,**weak_base_adjacents}
        else:
            raise ValueError("On a square lattice, adjacents must be one of ['edge'|'vertex'] connected")
    
        #return a dictionary of the neighbors for the input coordinate
        return {key: self.coord+value for key,value in base_adjacents.items()}

    @classmethod
    def valid_coord(cls,coord: np.array) -> bool:
        """all discrete cartesian coordinates are valid"""
        if super().valid_coord(coord) ==False:
            return False
        
        #check that the parameters are all ints (or floats which are equal to ints)
        for param in coord:
            if int(param)-param!=0:
                return False

        return True

