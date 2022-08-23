"""Utilities for performing geometric operations on matrices

rotate_cartesian:
    rotate an set of points counterclockwise around a point, by default the origin, on an axis

"""
from typing import Generator, Union
import numpy as np


def r_from_normal(angle, normal, discretise=True):
    """rotation matrix from a unit vector normal to the plane of rotation in discrete space
    Parameters:
    :param angle: float: angle to rotate by
    :param normal: np.ndarray: unit vector normal to the plane of rotation
    :return: np.ndarray: rotation matrix
    """
    try:
        normal = np.array(normal)
    except:
        raise ValueError('normal must be sequence')

    x, y, z = normal
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([
        [x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s],
        [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c]
    ])
    if discretise:
        return r.astype(int)
    else:
        return r



def rotate_normal(array: np.ndarray, turns: int, base_angle: float = np.pi/2, around: np.ndarray = None, axis: Union[str, np.array] = None, ) -> np.ndarray:
    """rotate an set of points counterclockwise around a point, by default the origin in discrete space
    (negatives are clockwise)

    Parameters:
    :param array: np.ndarray: array of points to rotate
    :param turns: int: number of turns to rotate by
    :param base_angle: float: angle to rotate by
    :param around: np.ndarray: point to rotate around
    :param axis: str: axis to rotate around or a vector normal to the plane of rotation

    :return: np.ndarray: rotated array

    for a description of the axis and angle rotation, see:
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """

    num_rots = 2*np.pi/base_angle
    # nothing about this function is inherently discrete, so this check might need to be removed should the library be expanded to real spaces
    if abs(num_rots-int(num_rots)) > 0.001:
        raise ValueError(
            f'the base angle should divide 2pi; 2pi/base_angle must be an integer')
    if type(turns) != int:
        raise ValueError('turns must be an integer')

    dims = array.shape[-1]
    theta = turns * base_angle
    if dims == 3:
        if type(axis) == str:  # generate a normal vector from a string representing an axis
            normal_dict = {
                'x': np.array([1, 0, 0]),
                'y': np.array([0, 1, 0]),
                'z': np.array([0, 0, 1])
            }
            norm = np.array([0, 0, 0])
            for c in ['x', 'y', 'z']:
                if c in axis:

                    norm += normal_dict[c]

            
        elif hasattr(axis, '__iter__'):  # axis is a vector: is already the normal
            norm = np.array(axis)
        else: # axis has to have a value in 3D as there is no such thing as a 'generic' 3D rotation, as rotations are inherently 2
            raise ValueError('axis must be specified for 3D arrays')

        if norm.tolist() not in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]:
                # hacky solution but cheap; if this method is extended to non-discrete spaces, this check should 
                # be removed
                raise ValueError('axis must either be a string representing the axis or a vector normal ')
        
        r = r_from_normal(theta, norm)

    else: # 2D: a rotation in 2D is simply the first two components of a rotation on the plane z=0, i.e. normal [0,0,1]
        r = r_from_normal(theta, [0, 0, 1])[:2, :2]

    if around is None:
        around = np.array([0, 0]) if dims == 2 else np.array([0, 0, 0])

    try:
        return (r@(array-around).T).T + around
    except ValueError as e:
        raise ValueError(
            'array and around must have the same trailing axes') from e



def isometric(array: np.ndarray) -> np.ndarray:
    """isometric transformation of triangular coordinates onto a 2D cartesian plane, for visualization purposes
    Parameters:
    :param array: np.ndarray: array of points to rotate
    :return: np.ndarray: rotated array


    https://en.wikipedia.org/wiki/Isometric_projection
    """

    xs, ys, zs = array.T
    us = (xs-zs)/np.sqrt(2)
    vs = (xs+2*ys+zs)/np.sqrt(6)
    return np.array([us, vs]).T

def cube_rotation_list(array: np.ndarray) -> Generator:
    """generator of 24 rotations in the regular octahedral rotation group S_4 for an array of 3D cartesian coordinates"""
    # based on https://stackoverflow.com/questions/16452383/how-to-get-all-24-rotations-of-a-3-dimensional-array
    # expanded for dealing with an array of coordinates
    def roll(arr) : return rotate_normal(arr,1,axis = np.array([0,1,0]))
    def turn(arr) : return rotate_normal(arr,1,axis =np.array([0,0,1])) # yaw
    def sequence(arr):
        for _side in range(2):
            for _roll in range(3):
                arr = roll(arr)
                yield arr
                for _turn in range(3):
                    arr = turn(arr)
                    yield arr
            arr = roll(turn(roll(arr)))

    return sequence(array)
