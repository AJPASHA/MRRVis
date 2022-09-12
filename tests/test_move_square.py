
"""tests for the Square moveset, and by extension the move class formalism

We will test the following moves:
    - mrrvis.square.slide
    - mrrvis.square.rotate
"""

import warnings
import pytest
import numpy as np
from mrrvis.configuration import  ConfigurationGraph, edge_connected, vert_connected, remove_vertices
from mrrvis.cell import Square
from mrrvis.movesets import squaremoves
from mrrvis.move import Checkwrapper, Move, Collision, CollisionCase, Transformation


def test_CollisionCase():
    arbitrary_coord = np.array([-46, 25])
    case = CollisionCase(np.array([[1,0]])+arbitrary_coord, np.array([[0,1],[1,1]])+arbitrary_coord)
    assert np.all(case.rotate(1, around=arbitrary_coord).empty == np.array([[0,1]])+arbitrary_coord)
    assert np.all(case.rotate(1, around=arbitrary_coord).full == np.array([[-1,0],[-1,1]])+arbitrary_coord)

    case = CollisionCase(np.array([[1,0]]), np.array([[0,1],[1,1]]))
    graph = ConfigurationGraph('Square', np.array([[0,0], [0,1],[1,1]]))
    assert case.evaluate_case(graph) == True

    
    #graph with occupied path
    graph = ConfigurationGraph('Square', np.array([[0,0], [1,0], [0,1], [1,1]]))
    assert case.evaluate_case(graph) == False

    #graph without a surface to slide on
    ConfigurationGraph('Square', np.array([0,0]))
    assert case.evaluate_case(graph) == False

def test_Collision():
    # example of a sliding move (or)
    case0 = CollisionCase(np.array([[1,0]]), np.array([[0,1],[1,1]]))
    case1 = CollisionCase(np.array([1,0]), np.array([[0,-1], [1,-1]]))
    collision = Collision([case0,case1], 'or')
    graph = ConfigurationGraph('Square', np.array([[0,0], [0,1],[1,1]]))
    assert collision.evaluate_feasible(graph) is True

    # example of a rotating move (xor)
    case0 = CollisionCase(np.array([[1,1]]), np.array([[0,1]]))
    case1 = CollisionCase(np.array([[1,1]]), np.array([[1,0]]))
    collision = Collision((case0,case1), 'xor')
    graph = ConfigurationGraph('Square', np.array([[0,0],[1,1]]))
    assert collision.evaluate_feasible(graph) is False
    graph = ConfigurationGraph('Square', np.array([[0,0],[0,1]]))
    assert collision.evaluate_feasible(graph) is True

def test_CheckWrapper():

    graph = ConfigurationGraph(Square, vertices=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]))
    wrapped_graph = Checkwrapper(graph)
    assert wrapped_graph.bind(vert_connected).get() is not None


    rm_vertices = np.array([[1,0],[1,1]])
    graph  = graph.remove_vertices(rm_vertices, check_connectivity=False)

    wrapped_graph = Checkwrapper(graph)
    assert wrapped_graph.bind(vert_connected)() is None

def test_slide_valid():
    graph = ConfigurationGraph(Square, np.array([[0, 0], [1, 0], [0, 1]]))
    move = squaremoves.slide(graph, 2, 'E')


    assert np.all(move().vertices == np.array([[0, 0], [1, 0], [1, 1]]))

def test_slide_invalid():
    graph = ConfigurationGraph(Square,np.array([[0, 0], [1, 0], [0, 1]]))

    with pytest.warns(UserWarning):

        # in this case, the move is invalid because of collision
        action = squaremoves.slide(graph, 0, 'E')
        assert action() is None

        # in this case, the move is invalid because of disconnection
        action  = squaremoves.slide(graph, 0, 'W')
        assert action() is None


#test some more valid moves to check that they work
def test_slide():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))

    with pytest.warns(UserWarning):
        #This would move the first module to the right which would be a collision
        assert squaremoves.slide(graph, 0, 'E')() is None
        #This would fail because the resulting graph would be disconnected
        assert squaremoves.slide(graph, 0, 'W')() is None

    # These are the only valid moves for this configuration
    assert squaremoves.slide(graph, 0, 'N')() == ConfigurationGraph(Square, np.array([[1,3],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    assert squaremoves.slide(graph, 5, 'W')() == ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[2,1]]))

    #test that the direct vertex reference works
    assert squaremoves.slide(graph, [1,2], 'N')() == ConfigurationGraph(Square, np.array([[1,3],[2,2],[2,3],[3,3],[3,2],[3,1]]))

def test_slide_wrong_direction():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    with pytest.raises(ValueError):
        squaremoves.slide(graph, 0, 'SE')()

def test_rotate():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]), connect_type='vertex')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # None cases
        assert squaremoves.rotate(graph, 0, 'NE')() is None #This should cause a collision with [2,3]
        assert squaremoves.rotate(graph, 0, 'NW')() is None #This should result in a disconnected graph
        assert squaremoves.rotate(graph, [2,2], 'NE')() is None #This should flout the xor collision rule

    # Valid moves
    assert np.all(squaremoves.rotate(graph, 1, 'SW')().vertices[1] == np.array([1,1]))
    # There has been a bug with the isomorphism checker for vertex connectivity
    assert squaremoves.rotate(graph, [1,2], 'SE')() == ConfigurationGraph(Square, np.array([[2,1],[2,2],[2,3],[3,3],[3,2],[3,1]]))
    


def test_line_move():
    graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))

    # action = squaremoves.slide_line(graph, 0, 'E') #move center line east once

    # assert np.all(action().vertices == np.array([[2,2],[3,2],[2,3],[3,3],[4,2],[3,1]]))

    # action = squaremoves.slide_line(graph, 0, 'N') #move a single module with the line move

    # assert np.all(action().vertices == np.array([[1,3],[2,2],[2,3],[3,3],[3,2],[3,1]]))

    action = squaremoves.slide_line(graph, 0, 'S') #disconnecting case
    # print(edge_connected(action()))
    assert action() is None

