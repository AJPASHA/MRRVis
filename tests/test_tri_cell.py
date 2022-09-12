from tabnanny import check
import pytest
from mrrvis.cell import Tri
import numpy as np
from .testutils import compare_np_dicts


def test_init():
    coord = np.array([1, 0, 0])
    cell = Tri(coord)
    assert cell.rotation_angle == 2*np.pi/3
    assert cell.n_parameters == 3
    assert cell.dimensions == 2
    assert cell.connectivity_types == {'edge', 'vertex'}
    assert cell.point_up(cell.coord) == False


def test_invalid_init():
    with pytest.raises(ValueError):  # ValueError: invalid coordinate
        Tri(np.array([1, 1, 1]))
    with pytest.raises(ValueError):  # ValueError: float input
        Tri(np.array([2.5, 7]))
    with pytest.raises(ValueError):  # ValueError: invalid matrix of coordinates
        Tri(np.array([[0, 0]]))


def test_point_down():
    assert Tri.point_up(np.array([10, -4, -5])) == False
    assert Tri.point_up(np.array([10, -4, -6])) == True


def test_base_adjacents():

    assert compare_np_dicts(
        Tri.adjacent_transformations('edge', np.array([1, 0, 0])),
        {'N': np.array([0,-1,0]), 'SW': np.array(
            [0,0,-1]), 'SE': np.array([-1, 0, 0])}
    )
    assert compare_np_dicts(
        Tri.adjacent_transformations('edge', np.array([0, 0, 0])),
        {'S': np.array([0, 1, 0]), 'NW': np.array([1, 0, 0]),
         'NE': np.array([0, 0, 1])}
    )

    assert compare_np_dicts(
        Tri.adjacent_transformations('vertex', np.array([1, 0, 0])),
        {'N': np.array([0, -1, 0]), 'SW': np.array([0, 0, -1]), 'SE': np.array([-1, 0, 0]),
         'S': np.array([-1, 1, -1]), 'NW': np.array([1, -1, -1]), 'NE': np.array([-1, -1, 1])}

    )


def test_adjacents():

    # face down example (edge)
    cell = Tri(np.array([-100, 60, 40]))

    assert compare_np_dicts(
        cell.adjacent_transformations('edge', np.array([0, 0, 0])),
        cell.adjacent_transformations('edge', True)
    )

    check_dict = {k: cell.coord+v for k,
                  v in cell.adjacent_transformations('edge', np.array([0, 0, 0])).items()}
    assert compare_np_dicts(
        cell.adjacents('edge'),
        check_dict
    )

    # face up example (edge)
    cell = Tri(np.array([-30000, 30000, 1]))
    check_dict = {k: cell.coord+v for k,
                  v in cell.adjacent_transformations('edge', np.array([1, 0, 0])).items()}
    assert compare_np_dicts(
        cell.adjacents('edge'),
        check_dict
    )

    # face down example (vertex)
    cell = Tri(np.array([-4000000, 3999999, 1]))
    check_dict = {k: cell.coord+v for k,
                  v in cell.adjacent_transformations('vertex', np.array([0, 0, 0])).items()}
    assert compare_np_dicts(
        cell.adjacents('vertex'),
        check_dict
    )

    # face up example (vertex)
    cell = Tri(np.array([-30000, 1, 30000]))
    check_dict = {k: cell.coord+v for k,
                  v in cell.adjacent_transformations('vertex', np.array([1, 0, 0])).items()}
    assert compare_np_dicts(
        cell.adjacents('vertex'),
        check_dict
    )


def test___getitem__():
    cell = Tri(np.array([1, 0, 0]))
    print(cell['N'])
    assert np.all(cell['N'] == np.array([0, -1, 0])+cell.coord)
    assert np.all(cell['SW'] == np.array([0, 0, -1])+cell.coord)
    assert np.all(cell['SE'] == np.array([-1, 0, 0])+cell.coord)
    assert np.all(cell['S'] == np.array([-1, 1, -1])+cell.coord)
    assert np.all(cell['NW'] == np.array([1, -1, -1])+cell.coord)
    assert np.all(cell['NE'] == np.array([-1, -1, 1])+cell.coord)

    coord = np.array([0, 0, 0])
    cell = Tri(coord)
    assert np.all(cell['S'] == np.array([0, 1, 0])+coord)
    assert np.all(cell['NW'] == np.array([1, 0, 0])+coord)
    assert np.all(cell['NE'] == np.array([0, 0, 1])+coord)
    assert np.all(cell['N'] == np.array([1, -1, 1])+coord)
    assert np.all(cell['SW'] == np.array([1, 1, -1])+coord)
    assert np.all(cell['SE'] == np.array([-1, 1, 1])+coord)
