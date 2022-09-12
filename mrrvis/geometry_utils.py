"""Utilities for performing geometric operations on matrices
defines the following functions

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


def r_from_normal(angle:float, normal:np.ndarray, discrete=True)-> np.ndarray:
    """obtain rotation matrix from a unit vector normal to the plane of rotation

    Parameters:
    :param angle: angle to rotate by in radians
    :param normal: the vector normal to the plane of rotation
    :param discrete: whether to discretise the rotation matrix (only use if the angle is definitel)
    returns:
    :return: rotation matrix
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
    if discrete:
        return r.astype(int)
    else:
        return r


def norm_from_str(axis: str) -> np.ndarray:
    """generate a normal vector from a string representing an axis
    params:
    :param axis: some combination of the characters ['x','y','z']
    returns:
    :return: normal vector
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

    Parameters:
    :param array: array of points to rotate
    :param turns: number of turns to rotate by
    :param base_angle: angle to rotate by
    :param around: point to rotate around
    :param axis: axis to rotate around either as a combination of ['x','y','z'] or as a vector normal to the desired plane of rotation
    returns:
    :return: the rotated array

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
        r = r_from_normal(theta, norm, discrete=False)
    else:
        r = r_from_normal(theta, [0, 0, 1], discrete=True)[:2, :2]


    transformed_array = ((r@(array-around).T).T + around)
    return np.around(transformed_array).astype(np.int16)




def cube_rotation_list(coords: np.ndarray) -> Generator:
    """generator of 24 rotations in the regular octahedral rotation group S_4 for an array of 3D cartesian coordinates

    params:
    :param coords: the array of coordinates to transform
    
    returns:
    :return: a generator for the 24 different discrete 90degree rotations of a 3D shape

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
    

    :param coords: the array of coordinates to rotate

    :return: a generator containing the 4 different rotations of the vertices
    """
    yield coords
    for _ in range(3):
        coords = rotate_normal(coords, 1)
        yield coords


def tri_rotation_list(coords: np.ndarray) -> Generator:
    """generator of the 3 rotations of a set of coordinates in a triangular lattice

    :param coords: the array of coordinates to rotate

    :return: a generator containing the 4 different rotations of the vertices
    """
    yield coords
    for _ in range(2):
        coords = rotate_normal(coords, 1, base_angle=2*np.pi/3, axis='xyz')
        yield coords


def hex_rotation_list(coords: np.ndarray) -> Generator:
    """generator of the 6 rotations of a set of coordinates in a hexagonal lattice
    
    :param coords: the array of coordinates to rotate

    :return: a generator containing the 4 different rotations of the vertices
    """
    yield coords
    for _ in range(5):
        coords = rotate_normal(coords, 1, base_angle=np.pi/3, axis=np.array([1,1,1]))

        yield coords
