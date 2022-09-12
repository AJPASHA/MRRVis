"""Defines the abstract class Cell and its 4 concrete subclasses:
    Square, Hex, Tri, and Cube.
    The Cell class's concrete types are the basis for generating the information for lattice operations 
    in the configuration Graph
"""

from abc import ABC, abstractmethod
from typing import Literal
import warnings
import numpy as np


class Cell(ABC):
    def __init__(self, coord: np.ndarray) -> None:
        """A Cell Object
        :param coord: the coordinate position of the cell
        :type coord: np.ndarray or list[int]
        :raise ValueError: if the coord is not in the lattice
        """
        coord = np.array(coord)

        # test that input coordinate is valid
        if not(self.valid_coord(coord)):
            raise ValueError('Coordinate must exist on the lattice plane')

        self.coord = coord.astype(int)

    @classmethod
    @abstractmethod
    def adjacent_transformations(cls, connectivity, point_up=None) -> dict:
        """make a dictionary of the neighbors to a hypothetical cell at the origin
        :param connectivity: str: the connectivity type for the lattice
        :param point_up: bool: (for triangles) whether this cell points up
        :return: the dictionary of adjacent transformations

        """
        pass

    @classmethod
    @property
    @abstractmethod
    def connectivity_types(self) -> set:
        """a list of the connectivity types supported by the cell
        :return: a set of connectivity types
        """
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
        note. valid for all space-filling lattices ('Square', 'Tri', 'Hex', 'Cube').
        """
        if cls.dimensions == 2:     
            connectivity = 'edge'
        if cls.dimensions == 3:
            connectivity = 'face'
        return np.pi/(len(cls.adjacent_transformations(connectivity, cls.point_up(np.array([0, 0, 0]))))/cls.dimensions)

    @classmethod
    def point_up(cls, coord=None) -> bool:
        """Returns True if the cell is pointing up, False if pointing down,
        only implemented for triangular cells."""
        return True

    def adjacents(self, connectivity=None) -> dict:
        """the neighbors of the cell
        :param connectivity: the connectivity type of the cell, 
        connectivity will default to vertex except for hexagons where edge is the only valid type
        :type connectivity: None or Literal['edge', 'vertex', 'face']
        :return: adjacents to the cell coordinate given the connectivity for all compass directions
        :rtype: dict[str, np.ndarray]
        """
        if connectivity is None:
            if 'vertex' in self.connectivity_types:
                connectivity = 'vertex'
            else:
                connectivity = 'edge'

        base_adjacents = self.adjacent_transformations(
            connectivity, self.point_up(self.coord))

        # return a dictionary of the neighbors for the input coordinate
        return {key: self.coord+value for key, value in base_adjacents.items()}

    @classmethod
    @abstractmethod
    def valid_coord(cls, coord: np.ndarray) -> bool:
        """Returns True if the coord is valid for the cell type
        :param coord: the coordinate to test
        :type coord: np.ndarray
        :return: the truth value of the coord's validity
        :rtype: bool
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
        """Obtain an item from the neighbor dictionary by cardinal direction
        :param key: the cardinal direction of the neighbor
        :return: the coordinate of the neighbor
        """

        try:
            return self.adjacents()[key.upper()]
        except KeyError as e:
            raise e

    @classmethod
    def compass(cls, connectivity='edge'):
        """The keys which are available for a given connectivity level
        :param connectivity: The connectivity rule for defining adjacency
        :return: a list of directions
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

    def __init__(self, coord: np.ndarray) -> None:
        """A Square cell at the given coordinate location
        :param coord: a sequence with 2 elements, (x,y), where x,y are integers
        :type coord: np.ndarray or other list-like structure
        :raise ValueError: if the coord is not in the lattice
        
        
        attributes:
            dimensions: the number of dimensions ==2
            n_parameters: the number of parameters ==2
            connectivity_types: the valid connectivity types =={'edge', 'vertex'}
            coord: the coordinate of the cell
            rotation_angle: the angle for discrete rotations ==pi/2
        methods:
            adjacents: returns a dictionary of adjacent cells to this cell, given the connectivity
            adjacent_transformations: returns a dictionary of adjacent cells to an arbitrary cell at the origin
            valid_coord: returns a bool corresponding to the validity of a coordinate
            compass: obtain the keys for the adjacents dictionary
        """
        super().__init__(coord)

    @staticmethod
    def adjacent_transformations(connectivity: Literal['edge', 'vertex'], *_) -> dict:
        """Returns a dictionary of transformations for a cell situated at the origin
        :param connectivity: the connectivity rule for adjacency, either 'edge' or 'vertex' adjacency
        :return: a dictionary of coordinates which are adjacent to the origin
        """
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
        """tests the validity of a coordinate
        
        :param coord: a sequence with 2 elements: (x,y)
        
        :return: a boolean signifying whether the coordinate exists within the cell lattice
        """
        if super().valid_coord(coord) is False:
            return False
        return True


class Tri(Cell):
    dimensions = 2
    n_parameters = 3
    connectivity_types = {'edge', 'vertex'}

    def __init__(self, coord: np.ndarray) -> None:
        """A triangular cell at the given coordinate location

        :param coord: a sequence with 2 elements, (x,y), where x,y are integers
        :type coord: np.ndarray or other list-like structure
        :raise ValueError: if the coord is not in the lattice

        
        attributes:
            dimensions: the number of dimensions ==2
            n_parameters: the number of parameters ==3
            connectivity_types: the valid connectivity types =={'edge', 'vertex'}
            coord: the coordinate of the cell
            rotation_angle: the angle for discrete rotations ==2pi/3
        methods:
            adjacents: returns a dictionary of adjacent cells to this cell, given the connectivity
            adjacent_transformations: returns a dictionary of adjacent cells to an arbitrary cell at the origin
            valid_coord: returns a bool corresponding to the validity of a coordinate
            compass: obtain the keys for the adjacents dictionary            
       """
        super().__init__(coord)

    @classmethod
    def point_up(cls, coord:np.ndarray=None) -> bool:
        """check that the cell is an upward facing triangle

        :param coord: a sequence with 3 elements (x,y,z) where x,y,z are integers

        :return: a boolean representing whether the cell is an upward pointing triangle (true) or not


        remark. The coordinate system used for triangular grids can be represented as a face connected 'staircase of cubes'
        in 3D cartesian space  (with the z axis inverted). If the triangular cell is pointing down then its centroid is on the plane x+y+z=1, 
        if it is facing up then it is on the plane x+y+z=0
        """
        if coord is None:
            raise ValueError('coord must be specified')
        if not(cls.valid_coord(coord)):
            warnings.warn('Coordinate must exist on the lattice planes')
            return None

        x, y, z = coord

        if x+y+z == 0:
            return True

        return False

    @classmethod
    def adjacent_transformations(cls, connectivity: Literal['edge', 'vertex'], point_up: bool) -> dict:
        """The transformations of a cell situated at the origin
        :param connectivity: the connectivity type of the lattice
        :param point_up: is true if the cell at the origin points up
        :return: a dictionary with the adjacent cells
        :rtype: dict['str', np.ndarray]

        """


        # the base adjacent transformations vary based on whether or not a cell has a point facing south
        if isinstance(point_up, np.ndarray):
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
        """tests that the coordinate is in the lattice


        :param coord: a sequence with 3 elements (x,y,z), where x,y,z are integers

        :return: a boolean stating whether the coordinate is valid
        
        valid coords must be in the plane x+y+z=0 or x+y+z=1"""

        # check that the parameters are of correct shape(inherited from Cell)
        if super().valid_coord(coord) is False:
            return False

        # check that the parameters of the coord sum to 0 or 1:
        x, y, z = coord
        if x+y+z not in [0, 1]:
            return False

        return True


class Hex(Cell):
    dimensions = 2
    n_parameters = 3
    connectivity_types = {'edge'}

    def __init__(self, coord: np.array) -> None:
        """A hexagonal cell at the given location
        :param coord: a sequence with 2 elements, (x,y), where x,y are integers
        :type coord: np.ndarray or other list-like structure
        :raise ValueError: if the coord is not in the lattice
        attributes:
            dimensions: the number of dimensions ==2
            n_parameters: the number of parameters ==3
            connectivity_types: the valid connectivity types =={'edge'}
            coord: the coordinate of the cell
            rotation_angle: the angle for discrete rotations ==pi/3
        methods:
            adjacents: returns a dictionary of adjacent cells to this cell, given the connectivity
            adjacent_transformations: returns a dictionary of adjacent cells to an arbitrary cell at the origin
            valid_coord: returns a bool corresponding to the validity of a coordinate
            compass: obtain the keys for the adjacents dictionary     
        """
        super().__init__(coord)

    @staticmethod
    # note, because hex only has one configuration setting, there is no need to pass in connectivity
    def adjacent_transformations(*_) -> dict:
        """The transformations of a cell situated at the origin
        :param connectivity: the connectivity type of the lattice
        :param point_up: is true if the cell at the origin points up
        :return: a dictionary with the adjacent cells
        :rtype: dict['str', np.ndarray]

        """
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
        """tests that the coordinate is in the lattice


        :param coord: a sequence with 3 elements (x,y,z), where x,y,z are integers

        :return: a boolean stating whether the coordinate is valid

        remark. On a hex grid, valid coords must be in the plane x+y+z=0"""

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

    def __init__(self, coord: np.array) -> None:
        """A cubic cell at the given coordinate
        :param coord: a sequence with 2 elements, (x,y), where x,y are integers
        :type coord: np.ndarray or other list-like structure
        :raise ValueError: if the coord is not in the lattice
        attributes:
            dimensions: the number of dimensions ==3
            n_parameters: the number of parameters ==3
            connectivity_types: the valid connectivity types =={'edge', 'vertex','face'}
            coord: the coordinate of the cell
            rotation_angle: the angle for discrete rotations ==pi/2
        methods:
            adjacents: returns a dictionary of adjacent cells to this cell, given the connectivity
            adjacent_transformations: returns a dictionary of adjacent cells to an arbitrary cell at the origin
            valid_coord: returns a bool corresponding to the validity of a coordinate
            compass: obtain the keys for the adjacents dictionary     
        """
        super().__init__(coord)

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """tests that the coordinate is in the lattice


        :param coord: a sequence with 3 elements (x,y,z), where x,y,z are integers

        :return: a boolean stating whether the coordinate is valid

        all discrete cartesian coordinates are valid"""
        if super().valid_coord(coord) is False:
            return False
        else:
            return True

    @staticmethod
    def adjacent_transformations(connectivity: Literal['edge', 'vertex','face'], *_) -> dict:
        """The transformations of a cell situated at the origin
        :param connectivity: the connectivity type of the lattice
        :param point_up: is true if the cell at the origin points up
        :return: a dictionary with the adjacent cells
        :rtype: dict['str', np.ndarray]

        """
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
