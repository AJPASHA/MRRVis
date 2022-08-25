"""A simple demonstration of the visualisation of a square configuration going through a series of moves"""

from mrrvis.configuration import ConfigurationGraph
from mrrvis.vistools import plot_square_config
from mrrvis.cell import Square
from mrrvis.movesets import squaremoves
import numpy as np

graph = ConfigurationGraph(Square, np.array([[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]))

plot_square_config(graph)

graph = squaremoves.slide(graph,0, 'N')()

plot_square_config(graph)

graph = squaremoves.rotate(graph,0, 'NE')()

plot_square_config(graph)


