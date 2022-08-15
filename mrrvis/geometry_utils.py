"""useful utilities for geometry

defines 2 key functions:
    rotate(array, turns, base_angle, axis) -> np.array: discrete rotation around an axis
    mirror(array, axis) -> np.array: reflection around an axis

"""
import numpy as np

#rotation matrices for 2D and 3D
_2D_rotation_matrix = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
]).astype(int)
_3Dx_rotation_matrix = lambda theta: np.array([
    [1, 0,              0], 
    [0, np.cos(theta),  -np.sin(theta)], 
    [0, np.sin(theta),  np.cos(theta)]
]).astype(int)

_3Dy_rotation_matrix = lambda theta: np.array([
    [np.cos(theta), 0,  np.sin(theta)],
    [0,             1,  0],
    [-np.sin(theta), 0, np.cos(theta)]
]).astype(int)

_3Dz_rotation_matrix = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta),  0],
    [0,             0,              1]
]).astype(int)

_2Dx_mirror_matrix = np.array([
    [1,0],
    [0,-1]
])
_2Dy_mirror_matrix = np.array([
    [-1,0],
    [0,1]
])
_3Dx_mirror_matrix = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,-1]
])
_3Dy_mirror_matrix = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,1]
])
_3Dz_mirror_matrix = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,-1]
])


def rotate(array: np.ndarray, turns: int, base_angle:float = np.pi/2, around:np.ndarray= None, axis: str=None, ) -> np.ndarray:
    """rotate an set of points counterclockwise around a point, by default the origin
    (negatives are clockwise)

    Parameters:
    :param array: np.ndarray: array of points to rotate
    :param turns: int: number of turns to rotate by
    :param base_angle: float: angle to rotate by
    :param around: np.ndarray: point to rotate around
    :param axis: str: axis to rotate around

    :return: np.ndarray: rotated array

    """

    num_rots = 2*np.pi/base_angle
    if abs(num_rots-int(num_rots)) > 0.001:
        raise ValueError(f'the base angle should divide 2pi; 2pi/{base_angle} must be an integer')
    
    if type(turns) != int:
        raise ValueError('turns must be an integer')
    dims = array.shape[-1]
    theta = turns * base_angle
    if dims == 3:
 
        if axis is None:
            raise ValueError('axis must be specified for 3D arrays')
        else:
            # because matrix multiplication is non commutative
            rotation_matrix_dict = {
                'x': _3Dx_rotation_matrix(theta),
                'y': _3Dy_rotation_matrix(theta),
                'z': _3Dz_rotation_matrix(theta)
            }
            f = False # flag for recursion
            for c in axis:
                try:
                    if not(f):
                        r = rotation_matrix_dict[c]
                    else:
                        r = rotation_matrix_dict[c]@(r)
                except KeyError as e:
                    raise KeyError(f'axis must be a combination of x, y, or z') from e

                f=True
    else:
        r =  _2D_rotation_matrix(theta)
    
    if around is not None:
        try:
            return (r@(array-around).T).T + around
        except ValueError as e:
            raise ValueError('array and around must have the same trailing axes') from e
    else:
        return (r@array.T).T
  


def mirror(array:np.ndarray, axis = str):
    """mirror an array around an axis"""

    #not fully implemented yet
    if array.shape[-1] == 2:
        mirror_matrix_dict = {
            'x': _2Dx_mirror_matrix,
            'y': _2Dy_mirror_matrix
        }
        f=False # flag for recursion

    elif array.shape[-1] == 3:
        mirror_matrix_dict = {
            'x': _3Dx_mirror_matrix,
            'y': _3Dy_mirror_matrix,
            'z': _3Dz_mirror_matrix
        }
        
    else:
        raise ValueError('array must be 2D or 3D')
    
    f=False # flag for recursion
    for c in axis:
        try:
            if not(f):
                m = mirror_matrix_dict[c]
            else:
                m = mirror_matrix_dict[c]@(m)
        except KeyError as e:
            if array.shape[-1]==2:
                raise KeyError(f'axis must be a combination of x or y') from e
            else:
                raise KeyError(f'axis must be a combination of x, y, or z') from e
    
    return (m@array.T).T


        

