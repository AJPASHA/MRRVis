
import pytest
import numpy as np
from mrrvis import Move, Square, ConfigurationGraph
from mrrvis.move import Transformation
from mrrvis.movesets.square import slide


def test_init():
    """
    Test the initialization of a move, in this case 'slide' from the default square moveset
    """
    graph = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [0, 1]]))
    move = slide(graph, 2, 'E')

    transformation_0 = move.transformations[0]
    assert transformation_0.location == 2
    assert np.all(transformation_0.translation == [1,0])
    #check that collisions are generating correctly
    case0 = transformation_0.collisions[0]
    case1 = transformation_0.collisions[1]
    assert np.all(case0.empty == np.array([1,0]))
    assert np.all(case0.empty == case1.empty)
    assert np.all(case0.full == np.array([[0,-1],[1,-1]]))
    assert np.all(case1.full == np.array([[0,1],[1,1]]))


    assert move() is not None






    # assert move() == ConfigurationGraph(Square, np.array([[0, 0], [1, 1], [0, 1]]))

    
# from mrrvis.graph import ModuleGraph

