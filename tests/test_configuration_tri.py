from mrrvis.cell import Cell,  Tri
from mrrvis.configuration import ConfigurationGraph, add_vertices, remove_vertices

import numpy as np
import pytest

rotation_matrix_2D =lambda r: np.array([[np.cos(r), -np.sin(r)], 
                                        [np.sin(r), np.cos(r)]])


def test_init():
    """Test the creation of a module graph object"""
    graph = ConfigurationGraph('Tri')
    assert graph.Cell == Tri

graph = ConfigurationGraph('Tri', np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]), connect_type='vertex')

def test_get_index():
    assert graph.get_index([0,0,1]) == 3
    with pytest.warns(UserWarning):
        assert graph.get_index([-1,0,1]) is None

def test_edges():

    for edge in graph.edges_from_i(0,'edge'):
        assert edge in [{0,1},{0,2},{0,3}]

    for edge in graph.E:
        assert edge in [{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}]

def test_connected():
    assert graph.is_connected('vertex')
    assert graph.is_connected('edge')

    unconnected_graph = ConfigurationGraph('Tri', np.array([[1,0,0],[0,1,0]]), connect_type='edge')
    assert not unconnected_graph.is_connected()

def test_add_vertices():
    add_graph = add_vertices(graph, np.array([[1,0,-1],[1,-1,0]]))
    assert np.all(add_graph.vertices == np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,-1,0],[1,0,-1]]))


def test_remove_vertices():

    rm_graph = remove_vertices(graph, np.array([[1,0,0]]))
    assert np.all(rm_graph.vertices == np.array([[0,0,0],[0,1,0],[0,0,1]]))



# def test_disconnect_rm():
#     with pytest.warns(UserWarning):
#         graph = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [1, 1]]))
#         graph = remove_vertices(graph, np.array([[1, 0]]))
#         assert np.all(graph.vertices == np.array([[0, 0], [1, 0], [1, 1]]))


# def test_equals():
#     graph1 = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [1, 1]]))
#     graph2 = ConfigurationGraph(Square, np.array([[3, 3], [4, 4], [4, 3]]))
#     assert graph1 == graph2

# def test_equals_rotate_1():
#     graph1 = ConfigurationGraph(Square, np.array([[1,1], [1,2], [2,1]]))

#     graph2 = ConfigurationGraph(Square, np.array([[-1,-1],[-1,-2],[-2,-1]]))
#     assert graph1 == graph2

# def test_equals_rotate_2():
#     """a slightly more complex example"""
#     graph1 = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))
#     rotation = rotation_matrix_2D(np.pi/2).astype(int)
#     g2_verts = (rotation.dot(graph1.V.T).T) + np.array([[27,15]])


#     graph2 = ConfigurationGraph(Square, g2_verts)
#     assert graph1 == graph2

# def test_in():
#     assert np.array([0,1]) in ConfigurationGraph(Square, np.array([[1,1],[0,1]]))
#     assert np.array([10,1]) not in ConfigurationGraph(Square, np.array([[1,1],[0,1]]))