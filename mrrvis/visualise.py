"""core visualisation tools for visualising configurations"""

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import math
import numpy as np
from mrrvis.configuration import ConfigurationGraph, min_coord, max_coord


def plot_square_config(configuration: ConfigurationGraph, show=True, save=False, filename=None):
    """
    Plot a square configuration
    """
    _, ax = plt.subplots(1, figsize=(5,5))
    ax.set_aspect('equal')

    vertices = configuration.vertices
    for vertex in vertices:
        patch = RegularPolygon(
            (vertex[0], vertex[1]), 
            numVertices=4, 
            radius = math.sqrt(0.5), 
            orientation=(math.pi/4)-1e-3, 
            ec='black', 
            lw=1, 
            ls='-',
        )
        ax.add_patch(patch)
    
    ax.scatter(vertices[:,0], vertices[:,1], alpha=0.3)
    plt.axis('off')
    if show:
        plt.show()
    if save:
        plt.savefig(filename)