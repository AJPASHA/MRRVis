import pytest
import numpy as np
from mrrvis.movesets import hexmoves
from mrrvis.configuration import ConfigurationGraph
def test_hex_rotate():
    vertices = [[-1,0,1],[1,-1,0],[0,0,0],[0,-1,1],[1,0,-1]]
    graph = ConfigurationGraph('Hex', vertices)
    action = hexmoves.rotate(graph, [0,0,0], 'NW')
    assert np.all(action().vertices ==[[-1,0,1],[1,-1,0],[-1,1,0],[0,-1,1],[1,0,-1]])
    # hexenv.step('rotate',[0,0,0],'NW')
