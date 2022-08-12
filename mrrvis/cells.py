"""This module contains the different subclasses of the Cell class."""

from .cell import Cell
import numpy as np


"""Square Lattice Cell"""
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
        elif connectivity == 'vertex':
            weak_adjacents = {
                'NE': strict_adjacents['N']+strict_adjacents['E'], 
                'NW': strict_adjacents['N']+strict_adjacents['W'], 
                'SE': strict_adjacents['S']+strict_adjacents['E'],
                'SW': strict_adjacents['S']+strict_adjacents['W']
            }
            return {**strict_adjacents, **weak_adjacents}
        else:
            raise ValueError(
                "On a square lattice, adjacents must be one of ['edge'|'vertex'] connected")

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """all discrete cartesian coordinates are valid"""
        if super().valid_coord(coord) == False:
            return False

        # check that the parameters are all ints (or floats which are equal to ints)
        for param in coord:
            if int(param)-param != 0:
                return False

        return True

"""Triangular Lattice Cell"""
class Tri(Cell):
    dimensions =2
    n_parameters = 2
    connectivity_types = {'edge', 'vertex'}

    def __init__(self, coord: np.array) -> None:
        """A triangular cell which can be considered as the prototype of the module in the module graph
        i.e. this informs the rest of the program what the shape of the cell is and how to calculate its neighbors"""
        super().__init__(coord)

    @staticmethod
    def point_down(coord) -> bool:
        """returns true if the cell is facing up
        The coordinate system used for triangular grids can be represented as a face connected 'staircase of cubes'
        in 3D cartesian space. If the triangular cell is pointing down then it is on the plane x+y+z=1 
        if it is facing up then it is on the plane x+y+z=0
        """
        x,y,z = coord
        if x+y+z == 0:
            return True
        else:
            return False

    @classmethod
    def adjacent_transformations(cls, connectivity, coord) -> dict:

        #the base adjacent transformations vary based on whether or not a cell has a point facing south
        if cls.point_down(coord):
            strict_adjacents = {'N': np.array([0,1,0]), 'SE': np.array([1,0,0]), 'SW': np.array([0,0,1])}
            if connectivity == 'vertex':
                weak_adjacents = {
                    'NE': strict_adjacents['N']-strict_adjacents['SE']+strict_adjacents['SW'], 
                    'NW': strict_adjacents['N']-strict_adjacents['SW']+strict_adjacents['SE'], 
                    'S': strict_adjacents['SE']-strict_adjacents['N']+strict_adjacents['SW']
                }
        else:
            strict_adjacents = {'NW': np.array([-1,0,0]), 'NE': np.array([0,0,-1]), 'S': np.array([0,-1,0])}
            if connectivity == 'vertex':
                weak_adjacents = {
                    'SW': strict_adjacents['S']-strict_adjacents['NE']+strict_adjacents['NW'], 
                    'SE': strict_adjacents['S']-strict_adjacents['NW']+strict_adjacents['NE'], 
                    'N': strict_adjacents['NE']-strict_adjacents['S']+strict_adjacents['NW']
                }

        if connectivity == 'edge':
            return strict_adjacents
        elif connectivity == 'vertex':
            return {**strict_adjacents, **weak_adjacents}
        else:
            raise ValueError('On a triangular lattice, adjacents must be one of ["edge"|"vertex"] connected')

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """valid coords must be in the plane x+y+z=0"""

        #check that the parameters are of correct shape(inherited from Cell)
        if super().valid_coord(coord) == False:
            return False

        # check that the parameters are all ints
        for param in coord:
            if int(param)-param != 0:
                return False

        # check that the parameters of the coord sum to 0 or 1:
        x,y,z = coord
        if x+y+z not in [0,1]:
            return False

        return True

"""Hexagonal Lattice Cell"""
class Hex(Cell):
    dimensions = 2
    n_parameters = 3
    connectivity_types = {'edge'}
    def __init__(self, coord: np.array) -> None:
        super().__init__(coord)
    
    @staticmethod
    def adjacent_transformations(*_) -> dict:
        """Returns a dictionary of transformations for this cell
        in a hex cell there are only edge connections        
        """
        return {
            'N': np.array([0,1,-1]), 
            'NE': np.array([1,0,-1]), 
            'SE': np.array([1,-1,0]) , 
            'S': np.array([0,-1,1]), 
            'SW': np.array([-1,0,1]),
            'NW': np.array([-1,1,0])
        }
    
    def valid_coord(self, coord: np.array) -> bool:
        """valid coords must be in the plane x+y+z=0"""

        #check that the parameters are of correct shape(inherited from Cell)
        if super().valid_coord(coord) == False:
            return False

        # check that the parameters are all ints
        for param in coord:
            if int(param)-param != 0:
                return False

        # check that the parameters of the coord sum to 0 or 1:
        x,y,z = coord
        if x+y+z != 0:
            return False

        return True

    

"""Cube Lattice Cell"""
class Cube(Cell):
    dimensions = 3
    n_parameters = 3
    connectivity_types = ['face', 'edge', 'vertex']

    def __init__(self, coord: np.array) -> None:
        super().__init__(coord)

    @classmethod
    def valid_coord(cls, coord: np.array) -> bool:
        """all discrete cartesian coordinates are valid"""
        if super().valid_coord(coord) == False:
            return False

        # check that the parameters are all ints (or floats which are equal to ints)
        for param in coord:
            if int(param)-param != 0:
                return False

        return True

    @staticmethod
    def adjacent_transformations(connectivity, _=None) -> dict:
        """Returns a dictionary of transformations for this cell
        in a cube cell there are face, edge and vertex connections      
        """
        face_adjacents = {
            'N': np.array([0,1,0]), 
            'S': np.array([0,-1,0]), 
            'E': np.array([1,0,0]), 
            'W': np.array([-1,0,0]), 
            'U': np.array([0,0,1]), 
            'D': np.array([0,0,-1])
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






    