from mrrvis.cells.cell import Cell
from mrrvis.graph.ModuleGraph import ModuleGraph
from mrrvis.cells.square import Square
import numpy as np
import pytest

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

def test_init_invalid_connectivity():
    """Test the creation of a module graph object with an invalid connectivity"""
    with pytest.raises(ValueError):
        ModuleGraph(Square, connectivity='invalid')

def test_edges_single_valid():
    """Test the implementation of a simple edge get"""
    graph = ModuleGraph(Square, vertices=np.array([[0,0],[0,1]]))

    assert graph.V.shape == (2, 2)
    # print('edges', graph.edges_from_i(0))
    assert graph.edges_from_i(1) == [{1,0}]

def test_edges_valid():
    """get the edges of a simple graph """
    graph = ModuleGraph(Square, vertices=np.array([[0,0],[0,1],[1,0],[1,1]]))
    assert graph.E == [{0,1},{0,2},{1,3},{2,3}]