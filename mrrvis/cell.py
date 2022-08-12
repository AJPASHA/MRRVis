"""The Prototype class for different cell types."""

from abc import ABC, abstractmethod
import numpy as np


class Cell(ABC):
    def __init__(self, coord: np.array) -> None:
        coord = np.array(coord)

        #test that input coordinate is valid
        if not(self.valid_coord(coord)):
            raise ValueError('Coordinate must exist on the lattice plane')

        
        self.coord = coord.astype(int)
    
    @classmethod
    @abstractmethod
    def adjacent_transformations(cls,connectivity, coord=None) -> dict:
        """Returns a dictionary of transformations for this cell type"""
        pass
    
    @classmethod
    @property
    @abstractmethod
    def connectivity_types(self) -> set:
        """a list of the connectivity types supported by the cell"""
        pass

    @classmethod
    @property
    @abstractmethod
    def n_parameters(cls) -> int:
        """The expected number of parameters"""
        pass

    @classmethod
    @property
    @abstractmethod
    def dimensions(cls) -> int:
        """The number of dimensions of the cell."""
        pass

    @classmethod
    @property
    def rotation_angle(cls) -> float:
        """The rotation angle of the cell.
        valid for all equilateral lattices.
        """
        if cls.dimensions == 2:
            connectivity = 'edge'
        if cls.dimensions ==3:
            connectivity = 'face'
        return np.pi/(len(cls.adjacent_transformations(connectivity,[0,0]))/cls.dimensions)

    # @abstractmethod
    def adjacents(self, connectivity: str = 'vertex') -> dict:
        """the neighbors of the cell"""
        print(connectivity)
        base_adjacents = self.adjacent_transformations(connectivity, self.coord)

        # return a dictionary of the neighbors for the input coordinate
        return {key: self.coord+value for key, value in base_adjacents.items()}

    @classmethod
    @abstractmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """Returns True if the coord is valid for the cell type"""
        coord = np.array(coord)

        #test shape of input
        if coord.shape[0] != cls.n_parameters:
            return False
        #test that input is int
        for param in coord:
            if int(param)-param !=0:
                return False
            


    def __getitem__(self,key:str):
        """Obtain an item from the neighbor dictionary by cardinal direction"""
        # key = str(key)
        try:
            return self.adjacents()[key.upper()]
        except KeyError as Kerror:
            print('key should be a cardinal direction')
            raise Kerror

    @classmethod
    def compass(cls,connectivity = 'edge'):
        return cls.adjacent_transformations(connectivity).keys()

    def __repr__(self) -> str:
        return f"{self.CellType}module @ {self.coord}"
    
    def __str__(self) -> str:
        return self.__repr__()


class CellReal(Cell):
    def __init__(self, coord: np.array) -> None:
        """abstract prototype for real valued cell representations"""
        raise NotImplementedError('CellReal is not implemented yet')
        super().__init__(coord)
    
    @abstractmethod
    def dimensions(self) ->set:
        """area bounded by the cell"""
        pass

    @abstractmethod
    def occludes(self, other)-> bool:
        """checks whether the cell occludes with another of the same type"""
        pass

    @staticmethod
    @abstractmethod
    def will_occlude(coord1:np.array, coord2: np.array):
        """static method for checking collision between two coordinates, useful for defining if a cell should be allowed in the graph"""
        pass


        