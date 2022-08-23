import pytest
from mrrvis.cell import Hex
import numpy as np
from mrrvis.testutils import compare_np_dicts

def test_init_valid():
    cell = Hex(np.array([1,0,-1]))

    assert cell.rotation_angle == np.pi/3
    assert cell.n_parameters == 3
    assert cell.dimensions == 2
    assert cell.connectivity_types == {'edge'}
    assert set(cell.compass(connectivity='edge')) == {'N','S','NE','NW','SE','SW'}

def test_invalid_init():
    with pytest.raises(ValueError): #float input
        Hex([2.5,7,3])
    with pytest.raises(ValueError): # incorrect shape
        Hex([0,0])
    with pytest.raises(ValueError): # invalid coordinate (not on plane)
        Hex([1,1,1])

def test_base_adjacents():

    assert compare_np_dicts( # there is only one possibility for hexagonal lattice
        Hex.adjacent_transformations(),
        {'N': np.array([0,1,-1]), 'S': np.array([0,-1,1]), 'NW': np.array([-1,1,0]), 'NE': np.array([1,0,-1]), 'SW': np.array([-1,0,1]), 'SE': np.array([1,-1,0])}
    )

def test_adjacents():
    cell = Hex(np.array([-100, 60, 40]))
    check_dict = {k: cell.coord+v for k,v in Hex.adjacent_transformations().items()}
    assert compare_np_dicts(cell.adjacents(), check_dict)

def test___get__():
    cell = Hex(np.array([350000, -355000, 5000]))
    translations = {'N': np.array([0,1,-1]), 'S': np.array([0,-1,1]), 'NW': np.array([-1,1,0]), 'NE': np.array([1,0,-1]), 'SW': np.array([-1,0,1]), 'SE': np.array([1,-1,0])}
    assert np.all(cell['N'] == cell.coord+translations['N'])
    assert np.all(cell['S'] == cell.coord+translations['S'])
    assert np.all(cell['NW'] == cell.coord+translations['NW'])
    assert np.all(cell['NE'] == cell.coord+translations['NE'])
    assert np.all(cell['SW'] == cell.coord+translations['SW'])
    assert np.all(cell['SE'] == cell.coord+translations['SE'])
    with pytest.raises(KeyError):
        cell['E']



