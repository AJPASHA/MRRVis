import pytest
import numpy as np
from mrrvis.geometry_utils import rotate, mirror

def test_rot_2D():
    arr = np.array([[5,0],[0,5]])

    assert np.all(rotate(arr,1) == np.array([[0,5],[-5,0]]))
    assert np.all(rotate(arr,2) == np.array([[-5,0],[0,-5]]))
    assert np.all(rotate(arr,3) == np.array([[0,-5],[5,0]]))

    assert np.all(rotate(arr,4) == rotate(arr,0))
    assert np.all(rotate(arr,5) == rotate(arr, 1))

    assert np.all(rotate(arr,-1) == rotate(arr, 3))
    assert np.all(rotate(arr,-2) == rotate(arr, 2))
    assert np.all(rotate(arr,-3) == rotate(arr, 1))
    assert np.all(rotate(arr,-4) == rotate(arr, 0))

    with pytest.raises(ValueError):
        rotate(arr, 2.2)

def test_rot_around():
    arr = np.array([2,2]) #2x1 array
    assert np.all(rotate(arr,1, around=[1,1]) == np.array([0,2]))


    arr = np.array([[2,2],[3,3]]) #2x2 array
    assert np.all(rotate(arr,2, around=[1,1]) == np.array([[0,0],[-1,-1]]))

    arr = np.array([[2,2,2],[3,3,3]]) #2x3 array
    assert np.all(rotate(arr, 2, around=[1,1,1], axis='x') == np.array([[2,0,0],[3,-1,-1]]))




def test_rot_single_item():
    arr = np.array([1,0])
    #because the function is defined for a matrix, it seems prudent to check that it works for a single array
    assert np.all(rotate(arr,1) == np.array([0,1]))

def test_wrong_axis():
    arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    with pytest.raises(KeyError):
        rotate(arr,1, axis='foo')

    with pytest.raises(ValueError):
        rotate(arr,1)

    #the axis should be irrelevant to 2D rotations
    arr = np.array([1,0])
    assert np.all(rotate(arr,1) == rotate(arr,1, axis='xy'))

def test_wrong_turns():
    arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    with pytest.raises(ValueError):
        rotate(arr,2.2, axis='xy')


def test_rot_3D():
    arr = np.array([[5,0,0],[0,5,0],[0,0,5]])
    print(rotate(arr,1,axis='x'))
    assert np.all(rotate(arr,1,axis='x') == np.array([[5,0,0],[0,0,5],[0,-5,0]]))
    assert np.all(rotate(arr,1,axis='y') == np.array([[0,0,-5],[0,5,0],[5,0,0]]))
    assert np.all(rotate(arr,1,axis='z') == np.array([[0,5,0],[-5,0,0],[0,0,5]]))

def test_rot_3D_composition():
    arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    #probably need some more tests here lol
    # assert np.all(rotate(arr,1,axis='yx') == rotate(arr,1,axis='xy'))
    assert np.all(rotate(arr,4, axis='xy') == arr)
    assert np.all(rotate(arr,4, axis='xyz')==arr)
    assert np.all(rotate(arr,4*5000, axis='xyz') == arr)
    assert np.any(rotate(arr,1, axis='xy') != rotate(arr,1, axis='yx'))
    assert np.all(rotate(arr,2, axis='xy') == rotate(arr,-2, axis='xy'))
# def test_mirror()


    
    