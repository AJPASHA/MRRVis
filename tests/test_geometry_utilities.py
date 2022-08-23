
import pytest
import numpy as np
from mrrvis.geometry_utils import rotate_normal, isometric,r_from_normal, cube_rotation_list


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

def test_rot_2D():
    arr = np.array([[5,0],[0,5]])

    assert np.all(rotate_normal(arr,1) == np.array([[0,5],[-5,0]]))
    assert np.all(rotate_normal(arr,2) == np.array([[-5,0],[0,-5]]))
    assert np.all(rotate_normal(arr,3) == np.array([[0,-5],[5,0]]))

    assert np.all(rotate_normal(arr,4) == rotate_normal(arr,0))
    assert np.all(rotate_normal(arr,5) == rotate_normal(arr, 1))

    assert np.all(rotate_normal(arr,-1) == rotate_normal(arr, 3))
    assert np.all(rotate_normal(arr,-2) == rotate_normal(arr, 2))
    assert np.all(rotate_normal(arr,-3) == rotate_normal(arr, 1))
    assert np.all(rotate_normal(arr,-4) == rotate_normal(arr, 0))

    with pytest.raises(ValueError):
        rotate_normal(arr, 2.2)

def test_rot_around():
    arr = np.array([2,2]) #2x1 array
    assert np.all(rotate_normal(arr,1, around=[1,1]) == np.array([0,2]))


    arr = np.array([[2,2],[3,3]]) #2x2 array
    assert np.all(rotate_normal(arr,2, around=[1,1]) == np.array([[0,0],[-1,-1]]))

    arr = np.array([[2,2,2],[3,3,3]]) #2x3 array
    assert np.all(rotate_normal(arr, 2, around=[1,1,1], axis='x') == np.array([[2,0,0],[3,-1,-1]]))




def test_rot_single_item():
    arr = np.array([1,0])
    #because the function is defined for a matrix, it seems prudent to check that it works for a single array
    assert np.all(rotate_normal(arr,1) == np.array([0,1]))

def test_wrong_axis():
    arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    with pytest.raises(ValueError):
        rotate_normal(arr,1, axis='foo')

    with pytest.raises(ValueError):
        rotate_normal(arr,1)

    #the axis should be irrelevant to 2D rotations
    arr = np.array([1,0])
    assert np.all(rotate_normal(arr,1) == rotate_normal(arr,1, axis='xy'))

def test_wrong_turns():
    arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    with pytest.raises(ValueError):
        rotate_normal(arr,2.2, axis='xy')


def test_rot_3D():
    arr = np.array([[5,0,0],[0,5,0],[0,0,5]])
    print(rotate_normal(arr,1,axis='x'))
    assert np.all(rotate_normal(arr,1,axis='x') == np.array([[5,0,0],[0,0,5],[0,-5,0]]))
    assert np.all(rotate_normal(arr,1,axis='y') == np.array([[0,0,-5],[0,5,0],[5,0,0]]))
    assert np.all(rotate_normal(arr,1,axis='z') == np.array([[0,5,0],[-5,0,0],[0,0,5]]))

def test_rot_3D_composition():
    arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    #probably need some more tests here lol
    # assert np.all(rotate(arr,1,axis='yx') == rotate(arr,1,axis='xy'))
    assert np.all(rotate_normal(arr,4, axis='xy') == arr)
    assert np.all(rotate_normal(arr,4, axis='xyz')==arr)
    assert np.all(rotate_normal(arr,4*5000, axis='xyz') == arr)
    assert np.all(rotate_normal(arr,2, axis='xy') == rotate_normal(arr,-2, axis='xy'))

    
# def test_mirror()
def test_rotation_matrices():
    angle = np.pi/2
    r1 = r_from_normal(angle, np.array([1,0,0])) 
    r2 = _3Dx_rotation_matrix(angle)

    assert np.all(r1 == r2)
    
    assert np.all(r_from_normal(angle,np.array([0,1,0])) == _3Dy_rotation_matrix(angle))
    assert np.all(r_from_normal(angle,np.array([0,0,1])) == _3Dz_rotation_matrix(angle))

    assert np.all(r_from_normal(angle, [1,1,0]) == np.add(r_from_normal(angle, [1,0,0]), r_from_normal(angle, [0,1,0])))

def test_cube_rot_generator():
    base_arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    lst = cube_rotation_list(base_arr)
    new = next(lst)
    check = rotate_normal(base_arr,1, axis='y')
    assert np.all(new == check)
    new = next(lst)
    check = rotate_normal(check, 1, axis='z')
    assert np.all(new == check)
    new = next(lst)
    check = rotate_normal(check,1,axis = 'z')
    assert np.all(new==check)
    new = next(lst)
    check = rotate_normal(check,1,axis='z')
    assert np.all(new==check)
    new = next(lst)
    check = rotate_normal(check,1,axis='y')
    assert np.all(new==check)

    arr = [arr for arr in cube_rotation_list(np.array([[1,0,0],[0,1,0],[0,0,1]]))]
    assert len(arr) == 24
    
