import pytest
from mrrvis.cell import Cube
import numpy as np
from .testutils import compare_np_dicts

def test_init():
    cell = Cube(np.array([1,2,3]))

    assert cell.rotation_angle == np.pi/2
    assert cell.n_parameters == 3
    assert cell.dimensions == 3
    assert set(cell.connectivity_types) == {'edge', 'face', 'vertex'}
    assert set(cell.compass(connectivity='face')) == {'N','S','E','W','U','D'}
    assert set(cell.compass(connectivity='edge'))-set(cell.compass(connectivity='face')) == {
        'NE','NW','SE','SW','UN','US','UE','UW','DS','DE','DW','DN'}

    assert set(cell.compass(connectivity='vertex'))-set(cell.compass(connectivity='edge')) == {
        'UNE','UNW','USE','USW','DNE','DNW','DSE','DSW'}

def test_adjacent_transformations():
    assert compare_np_dicts(Cube.adjacent_transformations('face'), {
        'N': np.array([0,1,0]),
        'S': np.array([0,-1,0]),
        'E': np.array([1,0,0]),
        'W': np.array([-1,0,0]),
        'U': np.array([0,0,1]),
        'D': np.array([0,0,-1]),
    })
    # check the edge adjacents
    check_dict = {k: Cube.adjacent_transformations('edge')[k] for k in 
        set(Cube.compass(connectivity='edge'))-set(Cube.compass(connectivity='face'))}
    assert compare_np_dicts(check_dict, {
        'NE': np.array([1,1,0]),
        'SE': np.array([1,-1,0]),
        'SW': np.array([-1,-1,0]),
        'NW': np.array([-1,1,0]),
        'UN': np.array([0,1,1]),
        'US': np.array([0,-1,1]),
        'UE': np.array([1,0,1]),
        'UW': np.array([-1,0,1]),
        'DN': np.array([0,1,-1]),
        'DS': np.array([0,-1,-1]),
        'DE': np.array([1,0,-1]),
        'DW': np.array([-1,0,-1]),
    })

    # check the vertex adjacents
    check_dict = {k: Cube.adjacent_transformations('vertex')[k] for k in
        set(Cube.compass(connectivity='vertex'))-set(Cube.compass(connectivity='edge'))}
    assert compare_np_dicts(check_dict, {
        'UNE': np.array([1,1,1]),
        'UNW': np.array([-1,1,1]),
        'USE': np.array([1,-1,1]),
        'USW': np.array([-1,-1,1]),
        'DNE': np.array([1,1,-1]),
        'DNW': np.array([-1,1,-1]),
        'DSE': np.array([1,-1,-1]),
        'DSW': np.array([-1,-1,-1]),
    })

def test_adjacents():
    coord = np.array([35632,4005,2445])
    cell = Cube(coord)
    assert np.all(cell['N'] == coord+np.array([0,1,0]))
    assert np.all(cell['S'] == coord+np.array([0,-1,0]))
    assert np.all(cell['E'] == coord+np.array([1,0,0]))
    assert np.all(cell['W'] == coord+np.array([-1,0,0]))
    assert np.all(cell['U'] == coord+np.array([0,0,1]))
    assert np.all(cell['D'] == coord+np.array([0,0,-1]))
    assert np.all(cell['NE'] == coord+np.array([1,1,0]))
    assert np.all(cell['SE'] == coord+np.array([1,-1,0]))
    assert np.all(cell['SW'] == coord+np.array([-1,-1,0]))
    assert np.all(cell['NW'] == coord+np.array([-1,1,0]))
    assert np.all(cell['UN'] == coord+np.array([0,1,1]))
    assert np.all(cell['US'] == coord+np.array([0,-1,1]))
    assert np.all(cell['UE'] == coord+np.array([1,0,1]))
    assert np.all(cell['UW'] == coord+np.array([-1,0,1]))
    assert np.all(cell['DN'] == coord+np.array([0,1,-1]))
    assert np.all(cell['DS'] == coord+np.array([0,-1,-1]))
    assert np.all(cell['DE'] == coord+np.array([1,0,-1]))
    assert np.all(cell['DW'] == coord+np.array([-1,0,-1]))
    assert np.all(cell['UNE'] == coord+np.array([1,1,1]))
    assert np.all(cell['UNW'] == coord+np.array([-1,1,1]))
    assert np.all(cell['USE'] == coord+np.array([1,-1,1]))
    assert np.all(cell['USW'] == coord+np.array([-1,-1,1]))
    assert np.all(cell['DNE'] == coord+np.array([1,1,-1]))
    assert np.all(cell['DNW'] == coord+np.array([-1,1,-1]))
    assert np.all(cell['DSE'] == coord+np.array([1,-1,-1]))
    assert np.all(cell['DSW'] == coord+np.array([-1,-1,-1]))
    
    