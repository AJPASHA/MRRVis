from mrrvis.cell import Cell, Square
from mrrvis.configuration import ConfigurationGraph, add_vertices, remove_vertices

import numpy as np
import pytest

rotation_matrix_2D =lambda r: np.array([[np.cos(r), -np.sin(r)], 
                                        [np.sin(r), np.cos(r)]])


def test_init():
    """Test the creation of a module graph object"""
    graph = ConfigurationGraph(Square)
    assert graph.Cell == Square

    assert graph.V.shape == (0, 2)
    assert all(graph.edges == np.array([]))


def test_init_invalid_cell():
    """Test the creation of a module graph object with an invalid cell by verifying that using the prototype Cell is invalid"""
    with pytest.raises(NotImplementedError):
        ConfigurationGraph(Cell)


def test_init_invalid_connectivity_type():
    """Test the creation of a module graph object with an invalid connectivity"""
    with pytest.raises(ValueError):
        ConfigurationGraph(Square, connect_type='invalid')

def test_init_negative_vertices():
    vertices = np.array([[0, 0], [0, -1]])
    graph = ConfigurationGraph(Square, vertices)
    print(graph.vertices)
    assert np.all(graph.vertices == vertices)


def test_edges_single_valid():
    """Test the implementation of a simple edge get"""
    graph = ConfigurationGraph(Square, vertices=np.array([[0, 0], [0, 1]]))

    assert graph.V.shape == (2, 2)

    assert graph.edges_from_i(1) == [{1, 0}]


def test_get_index():
    graph = ConfigurationGraph(Square, vertices=np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]))
    assert graph.get_index([0, 0]) == 0
    assert graph.get_index([1, 1]) == 3
    assert graph.get_index([2, 0]) == 4
    assert graph.get_index([2, 1]) == 5
    with pytest.warns(UserWarning):
        graph.get_index([3, 0])


def test_edges():
    """get the edges of a simple graph """
    graph = ConfigurationGraph(Square, vertices=np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]]))
    assert graph.E == [{0, 1}, {0, 2}, {1, 3}, {2, 3}]


def test_connected_true():
    graph = ConfigurationGraph(Square, vertices=np.array(
        [[0, 0], [0, 1], [0, 2], [1, 2]]))
    assert graph.is_connected() == True

# need to test more edge cases


def test_connected_vertex():
    graph = ConfigurationGraph(Square, vertices=np.array(
        [[0, 0], [1, 1]]), connect_type='vertex')
    assert graph.is_connected() == True


def test_connected_false():
    with pytest.warns(UserWarning):
        graph = ConfigurationGraph(Square, vertices=np.array([[0, 0], [10, 1]]))
        assert graph.is_connected() == False


def test_invalid_vertex_add():
    graph = ConfigurationGraph(Square, vertices=np.array([[0, 0], [0, 1]]))
    with pytest.warns(UserWarning):
        graph = add_vertices(graph, np.array([[1.5, 0]]))
        assert np.all(graph.vertices == np.array([[0, 0], [0, 1]]))


def test_add_single_vertex():
    graph = ConfigurationGraph(Square)
    # we need to test that this works with a 1D array
    graph = add_vertices(graph, np.array([0, 0]))
    assert np.all(graph.vertices == np.array([[0, 0]]))


def test_add_vertices():
    graph = ConfigurationGraph(Square, vertices=np.array([[0, 0], [0, 1]]))
    graph = add_vertices(graph, np.array([[1, 0], [1, 1]]))
    assert np.all(graph.vertices == np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))


def test_remove_single_vertex():
    graph = ConfigurationGraph(Square, vertices=np.array([[0, 0], [0, 1]]))
    graph = remove_vertices(graph, np.array([0, 0]))
    assert np.all(graph.vertices == np.array([[0, 1]]))


def test_remove_vertices():
    graph = ConfigurationGraph(Square, vertices=np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]]))
    graph = remove_vertices(graph, np.array([[0, 1], ]))
    assert np.all(graph.vertices == np.array([[0, 0], [1, 0], [1, 1]]))


def test_disconnect_rm():
    with pytest.warns(UserWarning):
        graph = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [1, 1]]))
        graph = remove_vertices(graph, np.array([[1, 0]]))
        assert np.all(graph.vertices == np.array([[0, 0], [1, 0], [1, 1]]))


def test_equals():
    graph1 = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [1, 1]]))
    graph2 = ConfigurationGraph(Square, np.array([[3, 3], [4, 4], [4, 3]]))
    assert graph1 == graph2

def test_equals_rotate_1():
    graph1 = ConfigurationGraph(Square, np.array([[1,1], [1,2], [2,1]]))

    graph2 = ConfigurationGraph(Square, np.array([[-1,-1],[-1,-2],[-2,-1]]))
    assert graph1 == graph2

def test_equals_rotate_2():
    """a slightly more complex example"""
    graph1 = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    rotation = rotation_matrix_2D(np.pi/2).astype(int)
    g2_verts = (rotation.dot(graph1.V.T).T) + np.array([[27,15]])


    graph2 = ConfigurationGraph(Square, g2_verts)
    assert graph1 == graph2

# def test_funcs_vertex_connectivity():