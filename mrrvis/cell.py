"""Defines the abstract class Cell and its 4 concrete subclasses:
    Square, Hex, Tri, and Cube.
    The Cell class's concrete types are the basis for generating the information for lattice operations 
    in the configuration Graph
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np


class Cell(ABC):
    def __init__(self, coord: np.array) -> None:
        coord = np.array(coord)

        # test that input coordinate is valid
        if not(self.valid_coord(coord)):
            raise ValueError('Coordinate must exist on the lattice plane')

        self.coord = coord.astype(int)

    @classmethod
    @abstractmethod
    def adjacent_transformations(cls, connectivity, point_up=None) -> dict:
        """Returns a dictionary of transformations for this cell type"""
        raise NotImplementedError('adjacent_transformations not implemented')

    @classmethod
    @property
    @abstractmethod
    def connectivity_types(self) -> set:
        """a list of the connectivity types supported by the cell"""
        raise NotImplementedError('connectivity_types not implemented')

    @classmethod
    @property
    @abstractmethod
    def n_parameters(cls) -> int:
        """The expected number of parameters"""
        raise NotImplementedError('n_parameters not implemented')

    @classmethod
    @property
    @abstractmethod
    def dimensions(cls) -> int:
        """The number of dimensions of the cell."""
        raise NotImplementedError('dimensions not implemented')

    @classmethod
    @property
    def rotation_angle(cls) -> float:
        """The rotation angle of the cell.
        valid for all equilateral lattices.
        """
        if cls.dimensions == 2:
            connectivity = 'edge'
        if cls.dimensions == 3:
            connectivity = 'face'
        return np.pi/(len(cls.adjacent_transformations(connectivity, cls.point_up(np.array([0,0,0]))))/cls.dimensions)

    @classmethod
    def point_up(cls, coord=None) -> bool:
        """Returns True if the cell is pointing up, False if pointing down,
        only implemented for triangular cells."""
        return coord

    def adjacents(self, connectivity=None) -> dict:
        """the neighbors of the cell"""
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
    def valid_coord(cls, coord: np.array) -> bool:
        """Returns True if the coord is valid for the cell type"""
        coord = np.array(coord)

        # test shape of input
        if coord.shape[0] != cls.n_parameters:
            return False
        # test that input is int
        for param in coord:
            if int(param)-param != 0:
                return False
        return True

    def __getitem__(self, key: str):
        """Obtain an item from the neighbor dictionary by cardinal direction"""

        try:
            return self.adjacents()[key.upper()]
        except KeyError as e:
            raise e

    @classmethod
    def compass(cls, connectivity='edge'):
        return cls.adjacent_transformations(connectivity).keys()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}module @ {self.coord}"

    def __str__(self) -> str:
        return self.__repr__()


class Square(Cell):
    dimensions = 2
    n_parameters = 2
    connectivity_types = {'edge', 'vertex'}

    def __init__(self, coord: np.array) -> None:
        """A square cell which can be considered as the prototype of the module in the module graph
        i.e. this informs the rest of the program what the shape of the cell is and how to calculate its neighbors
        """
        super().__init__(coord)

    @staticmethod
    def adjacent_transformations(connectivity, _=None) -> dict:
        """Returns a dictionary of transformations for this cell
        connectivity: if true, only returns facet (edge) connected neighbors, 
        otherwise will give both facet and edge connected neighbors
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
    def valid_coord(cls, coord: np.array) -> bool:
        """all discrete cartesian coordinates are valid"""
        if super().valid_coord(coord) is False:
            return False
        return True


class Tri(Cell):
    dimensions = 2
    n_parameters = 3
    connectivity_types = {'edge', 'vertex'}

    def __init__(self, coord: np.array) -> None:
        """A triangular cell which can be considered as the prototype of the module in the module graph
        i.e. this informs the rest of the program what the shape of the cell is and how to calculate its neighbors"""
        super().__init__(coord)

    @classmethod
    def point_up(cls, coord=None) -> bool:
        """returns true if the cell is pointing up
        The coordinate system used for triangular grids can be represented as a face connected 'staircase of cubes'
        in 3D cartesian space. If the triangular cell is pointing down then it is on the plane x+y+z=1 
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
    def adjacent_transformations(cls, connectivity, point_up) -> dict:

        # the base adjacent transformations vary based on whether or not a cell has a point facing south
        if isinstance(point_up, np.ndarray):
            point_up = cls.point_up(point_up)
        if point_up:
            strict_adjacents = {'NW': np.array(
                [-1, 0, 0]), 'NE': np.array([0, 0, -1]), 'S': np.array([0, -1, 0])}
            if connectivity == 'vertex':
                weak_adjacents = {
                    'SW': strict_adjacents['S']-strict_adjacents['NE']+strict_adjacents['NW'],
                    'SE': strict_adjacents['S']-strict_adjacents['NW']+strict_adjacents['NE'],
                    'N': strict_adjacents['NE']-strict_adjacents['S']+strict_adjacents['NW']
                }
        else:
            strict_adjacents = {'N': np.array([0, 1, 0]), 'SE': np.array(
                [1, 0, 0]), 'SW': np.array([0, 0, 1])}
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
        """valid coords must be in the plane x+y+z=0"""

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
        """A hexagonal cell which can be considered as the prototype of the module in the module graph
        i.e. this informs the rest of the program what the shape of the cell is and how to calculate its neighbors"""
        super().__init__(coord)

    @staticmethod
    def adjacent_transformations(*_) -> dict: # note, because hex only has one configuration setting, there is no need to pass in connectivity
        """Returns a dictionary of transformations for this cell
        in a hex cell there are only edge connections        
        """
        return {
            'N': np.array([0, 1, -1]),
            'NE': np.array([1, 0, -1]),
            'SE': np.array([1, -1, 0]),
            'S': np.array([0, -1, 1]),
            'SW': np.array([-1, 0, 1]),
            'NW': np.array([-1, 1, 0])
        }

    def valid_coord(self, coord: np.array) -> bool:
        """valid coords must be in the plane x+y+z=0"""

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
    connectivity_types = ['face', 'edge', 'vertex']

    def __init__(self, coord: np.array) -> None:
        super().__init__(coord)

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """all discrete cartesian coordinates are valid"""
        if super().valid_coord(coord) is False:
            return False
        else:
            return True

    @staticmethod
    def adjacent_transformations(connectivity, _=None) -> dict:
        """Returns a dictionary of transformations for this cell
        in a cube cell there are face, edge and vertex connections      
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

        raise ValueError('On a cube lattice, adjacents must be one of ["face"|"edge"|"vertex"] connected')