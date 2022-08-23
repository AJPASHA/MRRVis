"""Tests for the Cell class, instantiated as its square subclass

(because abstract classes are hard to unit test in themselves)
"""

import pytest
import numpy as np
from mrrvis.cell import Cell, Square

"""Square cell test cases"""
def test_init_valid():
    cell = Square(np.array([2, 1]))
    # assert np.all([cell.coord, np.array([2, 1])])
    assert cell.rotation_angle == np.pi/2
    assert cell.n_parameters == 2
    assert cell.dimensions == 2
    assert cell.connectivity_types == {'vertex','edge'}
    assert set(cell.compass(connectivity='edge')) == {'N','S','E','W'}

def test_invalid_init_float():
    with pytest.raises(ValueError):
        Square([2.5,7])
def test_invalid_init_len():
    with pytest.raises(ValueError):
        Square([1,1,1])

def test_invalid_2D():
    """test that invalid 2D coordinates are handled correctly"""
    with pytest.raises(ValueError):
        Square(np.array([[0, 0]]))


def test_invalid_subclass():
    """test that failure to implement abstract class methods raises error"""
    with pytest.raises(NotImplementedError):
        class InvalidCell(Cell):
            def __init__(self, coord: np.array) -> None:
                super().__init__(coord)
            def adjacents(self, connectivity: str) -> dict:
                pass
            def valid_coord(self, coord: np.array) -> bool:
                pass
        InvalidCell(np.array([0,0]))


def test___getitem__():
    cell = Square(np.array([2,1]))
    assert np.all(cell['N'] == np.array([2,2]))
    assert np.all(cell['S'] == np.array([2,0]))
    assert np.all(cell['E'] == np.array([3,1]))
    assert np.all(cell['W'] == np.array([1,1]))
    assert np.all(cell['NE'] == np.array([3,2]))
    assert np.all(cell['NW'] == np.array([1,2]))
    assert np.all(cell['SE'] == np.array([3,0]))
    assert np.all(cell['SW'] == np.array([1,0]))

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

#tri weak adjacents
# 'NE': np.array([1,1,-1]), 'NW': np.array([-1,1,1]), 'S': np.array([1,-1,1])
# 'SW': np.array([-1,-1,1]), 'SE': np.array([1,-1,-1]), 'N': np.array([-1,1,-1])