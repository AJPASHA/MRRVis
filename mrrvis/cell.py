"""The Prototype class for different cell types."""

from abc import ABC, abstractmethod
import numpy as np


class Cell(ABC):
    def __init__(self, coord: np.array) -> None:
        coord = np.array(coord)

        #test that input coordinate is valid
        if not(self.valid_coord(coord)):
            raise ValueError('Coordinate must exist on the lattice plane')
        
        self.coord = coord
    
    @property
    @abstractmethod
    def connectivity_types(self) -> set:
        """Return a list of the connectivity types supported by the cell"""
        pass


    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """The expected number of parameters"""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The number of dimensions of the cell."""
        pass

    @property
    def rotation_angle(self) -> float:
        """The rotation angle of the cell.
        valid for all equilateral lattices.
        This may have to be overridden in the case that a more complex (e.g. brevais) lattice is used
        """
        if self.dimensions == 2:
            connectivity = 'edge'
        if self.dimensions ==3:
            connectivity = 'face'
        
        return np.pi/(len(self.adjacents(connectivity))/self.dimensions)

    @abstractmethod
    def adjacents(self, connectivity: str) -> dict:
        """the neighbors of the cell"""
        pass

    @classmethod
    @abstractmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """Returns True if the coord is valid for the cell type"""
        coord = np.array(coord)

        #test shape of input
        if coord.shape[0] != cls.n_parameters:
            return False


    def __getitem__(self,key):
        """Obtain an item from the neighbor dictionary by cardinal direction"""
        key = str(key)
        try:
            return self.adjacents()[key.upper()]
        except KeyError as Kerror:
            print('key should be a cardinal direction')
            raise Kerror


    def dir_adjacents(self,connectivity = 'edge'):
        return self.adjacents(connectivity).keys()

    def __repr__(self) -> str:
        return f"{self.CellType}module @ {self.coord}"
    
    def __str__(self) -> str:
        return self.__repr__()


class CellReal(Cell):
    def __init__(self, coord: np.array) -> None:
        """abstract prototype for real valued cell representations"""
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


        