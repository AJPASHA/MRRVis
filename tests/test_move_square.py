
"""tests for the Square moveset, and by extension the move class formalism

We will test the following moves:
    - mrrvis.square.slide
    - mrrvis.square.rotate
"""

import pytest
import numpy as np
from mrrvis.configuration import  ConfigurationGraph
from mrrvis.cell import Square
from mrrvis.movesets import square


def test_init():
    """
    Test the initialization of a move, in this case 'slide' from the default square moveset
    """
    graph = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [0, 1]]))
    move = square.slide(graph, 2, 'E')

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
    
    assert hasattr(move, 'cell_type')
    assert hasattr(move, 'compass')
    assert hasattr(move, 'checklist')
    assert hasattr(move, 'collision_rule')

    assert move() is not None
    print('move is: ', move())
    assert np.all(move() == ConfigurationGraph(Square, np.array([[0,0],[1,0],[1,1]])))

#test some valid moves to check that they work
def test_slide():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))

    #This would move the first module to the right which would be a collision
    assert square.slide(graph, 0, 'E')() is None
    #This would fail because the resulting graph would be disconnected
    assert square.slide(graph, 0, 'W')() is None

    # These are the only valid moves for this configuration
    assert square.slide(graph, 0, 'N')() == ConfigurationGraph(Square, np.array([[1,3],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    assert square.slide(graph, 5, 'W')() == ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[2,1]]))

    #test that the direct vertex reference works
    assert square.slide(graph, [1,2], 'N')() == ConfigurationGraph(Square, np.array([[1,3],[2,2],[2,3],[3,3],[3,2],[3,1]]))

def test_slide_wrong_direction():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    with pytest.raises(ValueError):
        square.slide(graph, 0, 'SE')()

def test_rotate():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]), connect_type='vertex')

    # None cases
    assert square.rotate(graph, 0, 'NE')() is None #This should cause a collision with [2,3]
    assert square.rotate(graph, 0, 'NW')() is None #This should result in a disconnected graph
    assert square.rotate(graph, [2,2], 'NE')() is None #This should flout the xor collision rule

    # Valid moves
    assert np.all(square.rotate(graph, 1, 'SW')().vertices[1] == np.array([1,1]))
    # There has been a bug with the isomorphism checker for vertex connectivity
    assert square.rotate(graph, [1,2], 'SE')() == ConfigurationGraph(Square, np.array([[2,1],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    




