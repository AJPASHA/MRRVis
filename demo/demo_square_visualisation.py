"""A simple demonstration of the visualisation of a square configuration going through a series of moves"""

from mrrvis import ConfigurationGraph, Move, plot_square_config
from mrrvis.cells import Square
from mrrvis.movesets import square
import numpy as np

graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))

plot_square_config(graph)

graph = square.slide(graph,0, 'N')()

plot_square_config(graph)

graph = square.rotate(graph,0, 'NE')()

plot_square_config(graph)


