"""core visualisation tools for visualising configurations
provides the following functions:
    
hex_to_cart:
    convert hexagonal cubic coordinates to cartesian coordinates
tri_to_cart:
    convert triangular cubic coordinates to cartesian coordinates
square_patch_generator/tri_patch_generator/hex_patch_generator/cube_patch_generator:
    generators which produce a set of matplotlib patches for the given lattice type
plot_configuration:
    plot a configuration graph as a static plot
plot_history:
    plot a mrrvis.history.History object as an animation represented with jshtml 
"""

from typing import Generator, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from mrrvis.configuration import ConfigurationGraph
from mrrvis.history import History

from mrrvis.history import History

default_style = {   #These are the default settings for the patches produced by the plotting generators
    'ec':'black',
    'lw': 0,
    'ls': '-'
}
sqrt3 = np.sqrt(3)


def hex_to_cart(vertices: np.ndarray) -> np.ndarray:
    """convert an array of hexagon cubic coordinates to cartesian coordinates

    :param vertices: a set of hexagonal cubic coords expressed as a numpy array
    :return: an array of the converted vertices
    """
    x= vertices[:,0]
    y= 2.*np.sin(np.pi/3)*(vertices[:,1]-vertices[:,2])/3.
    # 2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3.
    return np.column_stack((x,y))

def tri_to_cart(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """convert an array of Triangle cubic coordinates to cartesian coordinates
    
    :param vertices: a set of hexagonal cubic coords expressed as a numpy array
    :return: an array of the converted vertices
    :return: a vector of whether each vertex(row) of vertices corresponds to an upward pointing triangle
    """
    point_up = np.apply_along_axis(lambda vi: True if sum(vi)==0 else False, axis=1, arr=vertices)

    xt,yt,zt = vertices[:,0], vertices[:,1], vertices[:,2]

    y = -yt*sqrt3  #y in cartesian space is the negative of triangle y, multiplied by sqrt3, the height of a triangle of side_length = 2
    y = np.array([y[i] + sqrt3/3 if not point_up[i] else y[i] for i in range(len(y))])  # if y points down, then we need to add sqrt3/3 so that the triangles are aligned along their sides
    x = xt-zt # xt and zt are the two diagonal directions, by subtracting the latter from the former, we get the horizontal distance as an integer
    

    vertices = np.column_stack((x,y))

    return vertices, point_up


def square_patch_generator(vertices: np.ndarray, **style) -> Generator[list, None, None]:
    """generator for square patches from a list of two dimensional vertices

    :param vertices: the vertices to generate patches for
    :param style: any matplotlib kwargs for styling the patches
    :return: a generator containing all of the patches
    """

        
    for vertex in vertices: # generates a 1x1 square polygon
        yield RegularPolygon(
            (vertex[0], vertex[1]), 
            numVertices=4, 
            radius = np.sqrt(0.5), # diagonal/2
            orientation=(np.pi/4)-1e-3, # floating point correction
            **style
        )

def hex_patch_generator(vertices: np.ndarray, **style) -> Generator[list, None, None]:
    """generator for hex patches from a list of two dimensional vertices
    :param vertices: the vertices to generate patches for
    :param style: any matplotlib kwargs for styling the patches
    :return: a generator containing all of the patches
    """
    vertices = hex_to_cart(vertices)

    for vertex in vertices:
        yield RegularPolygon(
            (vertex[0], vertex[1]), 
            numVertices=6,
            radius = 2/3,
            orientation=(np.pi/6),
            **style
        )


def tri_patch_generator(vertices:np.ndarray, **style) -> Generator[list, None,None]:
    """generator for tri patches from a list of two dimensional vertices
    :param vertices: the vertices to generate patches for
    :param style: any matplotlib kwargs for styling the patches
    :return: a generator containing all of the patches
    """
    vertices, point_up = tri_to_cart(vertices)

    for i in range(len(vertices)):
        points_up = point_up[i]
        orientation = 0 if points_up else np.pi #set orientation based on position
        h,v = vertices[i]
        
        yield RegularPolygon(
            (h,v), 
            numVertices=3, 
            radius = 2*sqrt3/3,
            orientation=orientation,
            **style
        )

def generate_voxels(vertices:np.ndarray)-> np.ndarray:
    """generate voxel array for cube visualisation
    
    :param vertices: the vertices to be voxelised
    :return: the voxelised array
    """

    X,Y,Z = vertices[:,0],vertices[:,1],vertices[:,2]
    x_range = np.min(X), np.max(X)
    y_range = np.min(Y), np.max(Y)
    z_range = np.min(Z), np.max(Z)

    shape = tuple(d_range[1]-d_range[0]+1 for d_range in (x_range, y_range,z_range))
    filled = np.zeros(shape, dtype=int)
    for v in vertices:
        x,y,z =  v
        x,y,z = x-np.min(X), y-np.min(Y),z-np.min(Z)
        filled[x,y,z] =1
    return filled
        


preprocessors = {   # these are the functions which convert the coordinate systems
    'Square': lambda X: X,
    'Cube': lambda X: X,
    'Hex': hex_to_cart,
    'Tri': tri_to_cart
}

patch_generators = { # associates cell types with patch generators
        'Square': square_patch_generator,
        'Hex': hex_patch_generator,
        'Tri': tri_patch_generator,
}


def plot_configuration(configuration: ConfigurationGraph,  show=True, save=False, filepath=None,axes=False, **style) -> None:
    """Plot a configuration with matplotlib
    params:
    :param configuration: the graph to visualise
    :param show: whether to show the resulting graph
    :param save: whether to save the resulting graph
    :param filepath: the filepath for the graph when saving
    :param axes: whether to show the axes
    :param style: any matplotlib kwargs    
    """
    if len(style) ==0:
        style = default_style

    
    cell_name = configuration.Cell.__name__
    vertices = configuration.vertices
    fig = plt.figure(figsize=(5, 5))
    if cell_name =='Cube': # generate a 3D plot from voxel matrix 
        ax = plt.axes(projection='3d')
        voxels = generate_voxels(vertices)
        ax.voxels(voxels, **style)
    else:   # generate a 2D plot from patches
        ax = fig.add_subplot()
        patches = patch_generators[cell_name](vertices, **style)
        
        [ax.add_patch(patch) for patch in patches]

        ax.set_aspect('equal')  # to avoid distortion of the patches
    ax.autoscale_view()
    if not axes:
        plt.axis('off') # we don't need to see the axes generally
    if show:
        plt.show()

    if save:
        plt.savefig(filepath)


def plot_history(history: History, speed:int= 200, show=True, save=False, filepath=None, **style) -> str:
    """Plot a history object as an animation with matplotlib: currently only works for 2D lattices (hex, square, tri)
    params:
    :param history: the history to animate
    :param speed: the length of each frame in milliseconds
    :param show: whether to show the resulting graph
    :param save: whether to save the resulting graph
    :param filepath: the filepath for the graph when saving
    :param axes: whether to show the axes
    :param style: any matplotlib kwargs    
    returns:
    :return: if show is set to true, then it returns an html representation of the video, 
    
    remark. in iPython, use iPython.display.HTML(plot_history(...,show=true,...)) to turn this into an inline video.
    """

    if len(style) == 0:
        style = default_style

    fig, ax = plt.subplots(1, figsize=(5, 5))

    ax.set_aspect('equal')
    plt.axis('off')

    cell_type = history.cell_type

    patches = patch_generators[cell_type](history[0].vertices, **style)
    collection = PatchCollection(list(patches))

    ax.add_collection(collection)

    def animate(frame):
        vertices = history[frame].vertices
        patches = patch_generators[cell_type](vertices, **style)
        collection.set_paths(list(patches))
        

        buff = 2 # the buffer for redrawing the limits
        ax.set_xlim(min(vertices[:,0])-buff, max(vertices[:,0])+buff) #adjusting limits to account
        ax.set_ylim(min(vertices[:,1])-buff, max(vertices[:,1])+buff)
        
        return collection


    anim = FuncAnimation(fig, animate, frames=len(history.history), interval=speed, blit=False, repeat=True)
    plt.close()
    if save:
        writer = PillowWriter(fps=24)
        anim.save(filepath, writer)
    if show:
        return anim.to_jshtml()


    


    

