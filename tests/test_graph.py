from mrrvis.cell import Cell
from mrrvis.graph import ModuleGraph
from mrrvis.cells import Square
import numpy as np
import pytest
import warnings

def test_init():
    """Test the creation of a module graph object"""
    graph = ModuleGraph(Square)
    assert graph.Cell == Square

    assert graph.V.shape == (0, 2)
    assert all(graph.edges == np.array([]))



def test_init_invalid_cell():
    """Test the creation of a module graph object with an invalid cell by verifying that using the prototype Cell is invalid"""
    with pytest.raises(TypeError):
        ModuleGraph(Cell)

def test_init_invalid_connectivity_type():
    """Test the creation of a module graph object with an invalid connectivity"""
    with pytest.raises(ValueError):
        ModuleGraph(Square, connect_type='invalid')

def test_edges_single_valid():
    """Test the implementation of a simple edge get"""
    graph = ModuleGraph(Square, vertices=np.array([[0,0],[0,1]]))

    assert graph.V.shape == (2, 2)
    # print('edges', graph.edges_from_i(0))
    assert graph.edges_from_i(1) == [{1,0}]

def test_edges():
    """get the edges of a simple graph """
    graph = ModuleGraph(Square, vertices=np.array([[0,0],[0,1],[1,0],[1,1]]))
    assert graph.E == [{0,1},{0,2},{1,3},{2,3}]

def test_connected_true():
    graph = ModuleGraph(Square, vertices=np.array([[0,0],[0,1],[0,2],[1,2]]))
    assert graph.is_connected() == True


def test_connected_vertex():
    graph = ModuleGraph(Square, vertices=np.array([[0,0],[1,1]]), connect_type='vertex')
    assert graph.is_connected() == True

def test_connected_false():
    with pytest.warns(UserWarning):
        graph = ModuleGraph(Square, vertices=np.array([[0,0],[10,1]]))
        assert graph.is_connected() == False

def test_add_vertex():
    graph = ModuleGraph(Square)
    graph.add_verts([[0,1],[1,1]])
    assert np.all(graph.vertices == np.array([[0,1],[1,1]]))

