"""The cell module defines the Cell abstract class and 4 subclasses:
Square, Hex, Tri and Cube.

The Cell subclasses define the shape and characteristics of a module in a lattice
bearing that name. For instance, if there is a square lattice then a cell.Square(coord)
object would know the location of adjacent cells, 
given a connectivity level ('edge' or 'vertex'), in this case assume 'edge',
and the 'compass' with which those neighbors are related to this cell, which for edge
connected square cells is ['N','W','S','E'].
"""

from abc import ABC, abstractmethod
from typing import Literal
import warnings
import numpy as np


class Cell(ABC):
    """A Cell class
    
    Parameters
    ----------
    coord: np.ndarray or list or tuple
        The coordinate of the cell

    Raises
    ------
    ValueError
        If coord is not a valid coordinate of the lattice used by
        this subclass
    
    Attributes
    ----------
    coord: np.ndarray
        The coordinate of the cell

    Methods
    -------
    __getitem__(key)
        access adjacent cell coordinates as keys of this object
    """
    def __init__(self, coord: np.ndarray) -> None:
        coord = np.array(coord)

        # test that input coordinate is valid
        if not(self.valid_coord(coord)):
            raise ValueError('Coordinate must exist on the lattice plane')

        self.coord = coord.astype(int)



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
        """The rotation angle of the cell."""

        if cls.dimensions == 2:     
            connectivity = 'edge'
        if cls.dimensions == 3:
            connectivity = 'face'

        if cls.__name__ == 'Tri':
            n_neighbors = len(cls.adjacent_transformations(connectivity, cls.point_up(np.array([0, 0, 0]))))
        else:
            n_neighbors = len(cls.adjacent_transformations(connectivity))
        return np.pi/(n_neighbors/cls.dimensions)
    
    @classmethod
    @abstractmethod
    def adjacent_transformations(cls, connectivity:str) -> dict:
        """Get the translations required to go to neighboring cells

        Parameters
        ----------
        connectivity: {'edge', 'vertex', 'face'}
            The level of connectivity required for a cell to be considered adjacent

        Returns
        -------
        dict of np.ndarrays
            the translation to the adjacent cell indexed by compass direction
        
        Raises
        ------
        ValueError
            if connectivity is not in this class.connectivity_types
        """
        pass


    def adjacents(self, connectivity=None) -> dict:
        """Get the neighbors of this cell
        
        Parameters
        ----------
        connectivity: {'edge', 'vertex', 'face'}, optional
            The level of connectivity required for a cell to be considered adjacent
            if left blank, will use the broadest set allowed: face>vertex>edge

        Returns
        -------
        dict of np.ndarrays
            A dictionary of adjacent cells indexed by compass directions
        """
        if connectivity is None:
            if 'face' in self.connectivity_types:
                connectivity='face'
            if 'vertex' in self.connectivity_types:
                connectivity = 'vertex'
            else:
                connectivity = 'edge'

        if self.__class__.__name__ =='Tri':
            base_adjacents = self.adjacent_transformations(
                connectivity, self.point_up(self.coord))
        else:
            base_adjacents = self.adjacent_transformations(connectivity)


        # return a dictionary of the neighbors for the input coordinate
        return {key: self.coord+value for key, value in base_adjacents.items()}

    @classmethod
    @abstractmethod
    def valid_coord(cls, coord: np.ndarray) -> bool:
        """Verify that a coordinate exists in the cell lattice

        Parameters
        ----------
        coord: np.ndarray or list or tuple
            The coordinate of the cell to check
        
        Returns
        -------
        bool
            Is True if the cell is a valid coord

        """
        coord = np.array(coord)

        # test shape of input
        if coord.shape[0] != cls.n_parameters:
            return False
        # test that input is int
        for param in coord:
            if int(param)-param != 0:
                return False
        return True

    def __getitem__(self, key: str)-> np.ndarray:
        """reference adjacents as keys of the cell object"""

        try:
            return self.adjacents()[key.upper()]
        except KeyError as e:
            raise e

    @classmethod
    def compass(cls, connectivity='edge'):
        """The keys which are available for a given connectivity level
        
        Parameters
        ----------
        connectivity: {'edge', 'vertex', 'face'}, default 'edge'
            The level of connectivity where a cell is considered adjacent
        
        Return
        ------
        list of str
            The list of compass directions for that level of connectivity
        """
        return cls.adjacent_transformations(connectivity).keys()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}module @ {self.coord}"

    def __str__(self) -> str:
        return self.__repr__()


class Square(Cell):
    dimensions = 2
    n_parameters = 2
    connectivity_types = {'edge', 'vertex'}

    @staticmethod
    def adjacent_transformations(connectivity: Literal['edge', 'vertex']) -> dict:
        strict_adjacents = {'N': np.array([0, 1]), 'S': np.array(
            [0, -1]), 'E': np.array([1, 0]), 'W': np.array([-1, 0])}

        if connectivity == 'edge':
            return strict_adjacents
        if connectivity == 'vertex':
            weak_adjacents = {
                'NE': strict_adjacents['N']+strict_adjacents['E'],
                'NW': strict_adjacents['N']+strict_adjacents['W'],
                'SE': strict_adjacents['S']+strict_adjacents['E'],
                'SW': strict_adjacents['S']+strict_adjacents['W']
            }
            return {**strict_adjacents, **weak_adjacents}

        raise ValueError(
            "On a square lattice, adjacents must be one of ['edge'|'vertex'] connected")

    @classmethod
    def valid_coord(cls, coord: np.ndarray) -> bool:
        if super().valid_coord(coord) is False:
            return False
        return True


class Tri(Cell):
    dimensions = 2
    n_parameters = 3
    connectivity_types = {'edge', 'vertex'}


    @classmethod
    def point_up(cls, coord:np.ndarray) -> bool:
        """Decide if the cell is upward_facing

        Only implemented for triangle cells
        
        Parameters
        ----------
        coord: np.ndarray
            The coord to check
        
        Returns
        -------
        bool
            returns true if and only if the triangle at this cell is upward facing
        
        Raises
        ------
        ValueError
            If the input coordinate is not valid
        """
        if not(cls.valid_coord(coord)):
            warnings.warn('Coordinate must exist on the lattice planes')
            return None

        x, y, z = coord

        if x+y+z == 0:
            return True

        return False

    @classmethod
    def adjacent_transformations(cls, connectivity: Literal['edge', 'vertex'], point_up: bool) -> dict:
        """Get the translations required to go to neighboring cells

        Parameters
        ----------
        connectivity: {'edge', 'vertex'}
            The level of connectivity required for a cell to be considered adjacent
        point_up: bool
            Is the cell an upward facing triangle
        
        Returns
        -------
        dict of np.ndarrays
            the translation to the adjacent cell indexed by compass direction
        
        Raises
        ------
        ValueError
            if connectivity is not in this class.connectivity_types
        """
        # the base adjacent transformations vary based on whether or not a cell has a point facing south
        if isinstance(point_up, np.ndarray):    #This allows a coordinate to be inputted
            point_up = cls.point_up(point_up)
        if point_up:
            strict_adjacents = {'NW': np.array(
                [1, 0, 0]), 'NE': np.array([0, 0, 1]), 'S': np.array([0, 1, 0])}
            if connectivity == 'vertex':
                weak_adjacents = {
                    'SW': strict_adjacents['S']-strict_adjacents['NE']+strict_adjacents['NW'],
                    'SE': strict_adjacents['S']-strict_adjacents['NW']+strict_adjacents['NE'],
                    'N': strict_adjacents['NE']-strict_adjacents['S']+strict_adjacents['NW']
                }
        else:
            strict_adjacents = {'N': np.array([0, -1, 0]), 'SE': np.array(
                [-1, 0, 0]), 'SW': np.array([0, 0, -1])}
            if connectivity == 'vertex':
                weak_adjacents = {
                    'NW': strict_adjacents['N']-strict_adjacents['SE']+strict_adjacents['SW'],
                    'NE': strict_adjacents['N']-strict_adjacents['SW']+strict_adjacents['SE'],
                    'S': strict_adjacents['SE']-strict_adjacents['N']+strict_adjacents['SW']
                }

        if connectivity == 'edge':
            return strict_adjacents
        if connectivity == 'vertex':
            return {**strict_adjacents, **weak_adjacents}

        raise ValueError(
            'On a triangular lattice, adjacents must be one of ["edge"|"vertex"] connected')

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """\n\nvalid coords must be in the plane x+y+z=0 or x+y+z=1"""
        __doc__ == super().valid_coord.__doc__ +__doc__ #append this subclass's conditions to parent documentation

        # check that the parameters are of correct shape(inherited from Cell)
        if super().valid_coord(coord) is False:
            return False

        # check that the parameters of the coord sum to 0 or 1:
        x, y, z = coord
        if x+y+z not in [0, 1]:
            return False

        return True
    
    @classmethod
    def compass(cls, point_up:bool, connectivity='edge')-> list:
        """The keys which are available for a given connectivity level
        
        Parameters
        ----------
        point_up: bool
            If True, get the compass of a north pointing tri, or a south pointing one otherwise
        connectivity: {'edge', 'vertex', 'face'}, default 'edge'
            The level of connectivity where a cell is considered adjacent
        
        Return
        ------
        list of str
            The list of compass directions for that level of connectivity
        """
        return cls.adjacent_transformations(connectivity, point_up).keys()


class Hex(Cell):
    dimensions = 2
    n_parameters = 3
    connectivity_types = {'edge'}

    @staticmethod
    
    def adjacent_transformations(*_) -> dict:
        # note, because hex only has one configuration setting, there is no need to pass in connectivity
        return {
            'N': np.array([0, 1, -1]),
            'NE': np.array([1, 0, -1]),
            'SE': np.array([1, -1, 0]),
            'S': np.array([0, -1, 1]),
            'SW': np.array([-1, 0, 1]),
            'NW': np.array([-1, 1, 0])
        }

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """\n\nremark. On a hex grid, valid coords must be in the plane x+y+z=0"""
        __doc__ == super().valid_coord.__doc__ +__doc__ #append this subclass's conditions to parent documentation

        # check that the parameters are of correct shape(inherited from Cell)
        if super().valid_coord(coord) is False:
            return False

        # check that the parameters of the coord sum to 0 or 1:
        x, y, z = coord
        if x+y+z != 0:
            return False

        return True


class Cube(Cell):
    dimensions = 3
    n_parameters = 3
    connectivity_types = {'face', 'edge', 'vertex'}

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        if super().valid_coord(coord) is False:
            return False
        else:
            return True

    @staticmethod
    def adjacent_transformations(connectivity: Literal['edge', 'vertex','face']) -> dict:

        face_adjacents = {
            'N': np.array([0, 1, 0]),
            'S': np.array([0, -1, 0]),
            'E': np.array([1, 0, 0]),
            'W': np.array([-1, 0, 0]),
            'U': np.array([0, 0, 1]),
            'D': np.array([0, 0, -1])
        }
        if connectivity == 'face':
            return face_adjacents

        edge_adjacents = {
            'NE': face_adjacents['N']+face_adjacents['E'],
            'SE': face_adjacents['S']+face_adjacents['E'],
            'SW': face_adjacents['S']+face_adjacents['W'],
            'NW': face_adjacents['N']+face_adjacents['W'],
            'UN': face_adjacents['U']+face_adjacents['N'],
            'US': face_adjacents['U']+face_adjacents['S'],
            'UE': face_adjacents['U']+face_adjacents['E'],
            'UW': face_adjacents['U']+face_adjacents['W'],
            'DN': face_adjacents['D']+face_adjacents['N'],
            'DS': face_adjacents['D']+face_adjacents['S'],
            'DE': face_adjacents['D']+face_adjacents['E'],
            'DW': face_adjacents['D']+face_adjacents['W']
        }
        if connectivity == 'edge':
            return {**face_adjacents, **edge_adjacents}

        vertex_adjacents = {
            'UNE': face_adjacents['N']+face_adjacents['E']+face_adjacents['U'],
            'USE': face_adjacents['S']+face_adjacents['E']+face_adjacents['U'],
            'UNW': face_adjacents['N']+face_adjacents['W']+face_adjacents['U'],
            'USW': face_adjacents['S']+face_adjacents['W']+face_adjacents['U'],
            'DNE': face_adjacents['N']+face_adjacents['E']+face_adjacents['D'],
            'DSE': face_adjacents['S']+face_adjacents['E']+face_adjacents['D'],
            'DSW': face_adjacents['S']+face_adjacents['W']+face_adjacents['D'],
            'DNW': face_adjacents['N']+face_adjacents['W']+face_adjacents['D']
        }
        if connectivity == 'vertex':
            return {**face_adjacents, **edge_adjacents, **vertex_adjacents}

        raise ValueError(
            'On a cube lattice, adjacents must be one of ["face"|"edge"|"vertex"] connected')
