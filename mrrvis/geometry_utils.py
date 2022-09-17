"""The geometry_utils module provides utilities for performing geometric operations on matrices

r_from normal:
    identify the rotation matrix of a normalised vector
norm_from_str:
    generate a norm from a string representing the cartesian axes that the rotation is in
rotate_normal:
    rotate an array of vertices using angle and axis rotation

cube_rotation_list/square_rotation_list/hex_rotation_list/tri_rotation_list:
    generators for the rotation groups of different shapes in a given lattice type
"""
from typing import Generator, Union
import numpy as np


def r_from_normal(angle:float, normal:np.ndarray, round=True)-> np.ndarray:
    """obtain rotation matrix from a unit vector normal to the plane of rotation

    Parameters
    ----------
    angle: float
        angle to rotate by in radians
    normal: ndarray
        the vector normal to the plane of rotation
    round: bool
        round the rotation matrix to Four decimal places if true; to avoid floating-point errors
        at an expense to accuracy for irregular angles
    
    Returns
    -------
    ndarray
        a rotation matrix for the given axis and angle
    """
    normal = np.array(normal)
    if normal.shape == ():
        raise ValueError('normal must be specified')

    x, y, z = normal
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([
        [x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s],
        [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c]
    ])
    if round:
        return np.around(r, 4)
    else:
        return r


def norm_from_str(axis: str) -> np.ndarray:
    """generate a normal vector from a string representing an axis

    Parameters
    ----------
    axis: str
        some combination of the characters ['x','y','z']
    
    Raises
    ------
    ValueError
        If non of the characters in axis are in ['x','y','z']
    
    Returns
    -------
    ndarray
        a vector pointing in the positive direction for the given axis
    
    """
    axis = axis.lower()
    normal_dict = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    norm = np.array([0, 0, 0])
    for c in ['x', 'y', 'z']:
        norm += normal_dict[c] if c in axis else 0
    if np.all(norm == 0):
        raise ValueError(
            'axis must either be a string representing the axis or a vector normal ')

    return norm


def rotate_normal(array: np.ndarray, turns: int, base_angle: float = np.pi/2, around: np.ndarray = None, axis: Union[str, np.array] = None, ) -> np.ndarray:
    """rotate an set of points counterclockwise around a point, by default the origin in discrete space

    Parameters
    ----------
    array: ndarray
        array of points to rotate
    turns: int
        number of turns to rotate by
    base_angle: float
        angle to rotate by
    around: ndarray
        point to rotate around
    axis: str or ndarray
        axis to rotate around either as a combination of ['x','y','z'] or as a vector normal to the desired plane of rotation
    
    Returns
    -------
    ndarray
        the rotated array

    Notes
    -----
    for a description of axis and angle rotation, see:
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    array = np.array(array)
    dims = array.shape[-1]
    theta = turns * base_angle
    num_rots = 2*np.pi/base_angle
    if abs(num_rots-int(num_rots)) > 0.001:
        raise ValueError(
            f'the base angle should divide 2pi; 2pi/base_angle must be an integer')
    if type(turns) != int:
        raise ValueError('turns must be an integer')

    # set around to the origin if not specified
    around = (np.array([0, 0]) if dims == 2 else np.array(
        [0, 0, 0])) if around is None else np.array(around)

    if around.shape[-1] != dims:
        raise ValueError(
            'around must be of equal dimension to array')

    if axis is None:

        if abs(base_angle % (np.pi/3)) <1e-3:

            axis= np.array([1,1,1])


    if dims == 3:
        norm = norm_from_str(axis) if type(axis) == str else np.array(axis)
        norm = norm / np.linalg.norm(norm)
        r = r_from_normal(theta, norm, round=False)
    else:
        r = r_from_normal(theta, [0, 0, 1], round=True)[:2, :2]


    transformed_array = ((r@(array-around).T).T + around)
    return np.around(transformed_array).astype(np.int16)




def cube_rotation_list(coords: np.ndarray) -> Generator:
    """generator of 24 rotations in the regular octahedral rotation group S_4 for an array of 3D cartesian coordinates

    Parameters
    ----------
    coords: ndarray
        the array of coordinates to transform
    
    Yields
    -------
    ndarray
        a rotation of the shape formed by the input coordinates

    Notes
    -----
    algorithm based on https://stackoverflow.com/questions/16452383/how-to-get-all-24-rotations-of-a-3-dimensional-array, 
        although expanded to deal with multiple coordinates at once
    """
    
    def roll(arr): return rotate_normal(
        arr, 1, axis=np.array([0, 1, 0]))  # roll

    def turn(arr): return rotate_normal(
        arr, 1, axis=np.array([0, 0, 1]))  # yaw

    def sequence(arr):
        for _ in range(2):  # side
            for _ in range(3):  # roll
                arr = roll(arr)
                yield arr
                for _ in range(3):  # turn
                    arr = turn(arr)
                    yield arr
            # switch 'side' so that the other 3 sides can be 'face up'
            arr = roll(turn(roll(arr)))

    return sequence(coords)


def square_rotation_list(coords: np.ndarray) -> Generator:
    """generator of the 4 rotations of a set of coordinates in a square lattice

    Parameters
    ----------
    coords: ndarray
        the array of coordinates to transform
    
    Yields
    -------
    ndarray
        a rotation of the shape formed by the input coordinates
    """
    yield coords
    for _ in range(3):
        coords = rotate_normal(coords, 1)
        yield coords


def tri_rotation_list(coords: np.ndarray) -> Generator:
    """generator of the 3 rotations of a set of coordinates in a triangular lattice

    Parameters
    ----------
    coords: ndarray
        the array of coordinates to transform
    
    Yields
    -------
    ndarray
        a rotation of the shape formed by the input coordinates
    """
    yield coords
    for _ in range(2):
        coords = rotate_normal(coords, 1, base_angle=2*np.pi/3, axis='xyz')
        yield coords


def hex_rotation_list(coords: np.ndarray) -> Generator:
    """generator of the 6 rotations of a set of coordinates in a hexagonal lattice
    
    Parameters
    ----------
    coords: ndarray
        the array of coordinates to transform
    
    Yields
    -------
    ndarray
        a rotation of the shape formed by the input coordinates
    """
    yield coords
    for _ in range(5):
        coords = rotate_normal(coords, 1, base_angle=np.pi/3, axis=np.array([1,1,1]))

        yield coords
