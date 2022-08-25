"""core visualisation tools for visualising configurations"""

from typing import Generator
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib import animation
import numpy as np
from mrrvis.configuration import ConfigurationGraph, min_coord, max_coord
from mrrvis.history import History

# def plot_discrete(configuration: ConfigurationGraph, show=True, save=False, filename=None) -> None:
#     """
#     Plot a discrete configuration
#     """
#     plotting_funcs = {
#         'Square': plot_square_config,
#         # 'Cube': plot_cube_config,
#         # 'Triangle': plot_triangle_config,
#         # 'Hexagon': plot_hexagon_config,
#     }
#     plotting_funcs[configuration.Cell.__name__](configuration, show, save, filename)
    
# def plot_square_config(configuration: ConfigurationGraph, show=True, save=False, filename=None):
#     """
#     Plot a square configuration
#     """
#     _, ax = plt.subplots( figsize=(5,5))
#     ax.set_aspect('equal')

#     vertices = configuration.vertices
#     for vertex in vertices:
#         patch = RegularPolygon(
#             (vertex[0], vertex[1]), 
#             numVertices=4, 
#             radius = np.sqrt(0.5), 
#             orientation=(np.pi/4)-1e-3, 
#             ec='black', 
#             lw=1, 
#             ls='-',
#         )
#         ax.add_patch(patch)
    
#     ax.scatter(vertices[:,0], vertices[:,1], alpha=0.3)
#     plt.axis('off')
#     if show:
#         plt.show()
#     if save:
#         plt.savefig(filename)

def square_patch_generator(vertices: np.ndarray, **style) -> Generator[list, None, None]:
    """generator for square patches from a list of two dimensional vertices"""
    for vertex in vertices:
        yield RegularPolygon(
            (vertex[0], vertex[1]), 
            numVertices=4, 
            radius = np.sqrt(0.5), # diagonal/2
            orientation=(np.pi/4)-1e-3, 
            **style
            # ec='black', 
            # lw=1, 
            # ls='-',
        )

def circle_patch_generator(vertices: np.ndarray) -> Generator[list, None, None]:
    for vertex in vertices:
        yield Circle(
            (vertex[0], vertex[1]), 
            radius = 0.5,
            ec='black', 
            lw=1, 
            ls='-',
        )


