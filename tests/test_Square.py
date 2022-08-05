import pytest
# from mrrvis import mrrvis
from mrrvis.cells import Square
import numpy as np

def test_init_valid():
    cell = Square(np.array([2, 1]))
    # assert np.all([cell.coord, np.array([2, 1])])
    assert cell.rotation_angle == np.pi/2
    assert cell.n_parameters == 2
    assert cell.dimensions == 2
    assert cell.connectivity_types == {'vertex','edge'}
    assert set(cell.dir_adjacents(connectivity='edge')) == {'N','S','E','W'}

def test_invalid_init_float():
    with pytest.raises(ValueError):
        Square([2.5,7])
def test_invalid_init_len():
    with pytest.raises(ValueError):
        Square([1,1,1])

def test___getitem__():
    cell = Square(np.array([2,1]))
    assert np.all(cell['N'] == np.array([2,2]))

    with pytest.raises(KeyError):
        cell['foo']


def test_valid_coord():
    Cell = Square
    assert Cell.valid_coord([1,1]) == True
    assert Cell.valid_coord([1000000,50000]) ==True
    assert Cell.valid_coord([0,0.5]) ==False


def test_neighbors_std():
    """test that neighbors are correct for a normal input"""
    cell = Square(np.array([3, 1]))
    adjacents = cell.adjacents()
    print (adjacents)
    expected_adjacents = {'N': np.array([3, 2]), 'S': np.array([3, 0]), 'E': np.array([4, 1]), 'W': np.array([2,1]),
                            'NE': np.array([4,2]), 'NW': np.array([2,2]), 'SE': np.array([4,0]), 'SW': np.array([2,0])}
    for key, value in expected_adjacents.items():
        assert np.all(value == adjacents[key])

def test_neighbors_strict():
    """test that strict adjacents works"""
    cell = Square(np.array([0, 0]))
    adjacents = cell.adjacents(connectivity='vertex')
    expected_adjacents = {'N': np.array([0, 1]), 'S': np.array([0, -1]), 'E': np.array([1, 0]), 'W': np.array([-1, 0])}
    for key, value in expected_adjacents.items():
        assert np.all(value == adjacents[key])

def test_neigbors_negative():
    """test that negative coordinates are handled correctly"""
    cell = Square(np.array([-1, -1]))
    adjacents = cell.adjacents('vertex')
    expected_adjacents = {'N': np.array([-1, 0]), 'S': np.array([-1, -2]), 'E': np.array([0, -1]), 'W': np.array([-2, -1])}
    for key, value in expected_adjacents.items():
        assert np.all(value == adjacents[key])
