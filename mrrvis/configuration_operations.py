"""
contains methods for module graph manipulation

1. Add vertices: generate a new graph with the vertices added
2. Remove vertices: generate a new graph with the vertices removed
3. Move vertices: generate a new graph with the vertices moved
3. equals: check if two graphs are equal
4. Connected: check if the graph is connected
5. get_index: get the index of a vertex in a set of vertices

"""


from mrrvis.configuration import ConfigurationGraph, connected, get_index
# from mrrvis.cells import Square, Hex, Tri, Cube
from mrrvis.move import Move
import numpy as np
import warnings


def add_vertices(graph: ConfigurationGraph, in_vertices: np.array, check_connectivity=True) -> ConfigurationGraph:
    """
    Add vertices to the graph

    parameters:
    :param graph: the graph to add vertices to
    :param in_vertices: the vertices to add
    :param check_connectivity: if true, check if the resulting graph is connected
    :return: the resulting graph


    In order for vertices to be valid, they must:
    1. be of the correct shape,
    2. be valid coordinates.
    Additionally, 
    3. if the graph is required to be connected, then the vertices must form a connected graph when concatenated
        with existing vertices.
    """
    Cell = graph.Cell
    in_vertices = np.array(in_vertices)

    #1. check that the coordinates are of the correct shape
    if in_vertices.shape[-1] != Cell.n_parameters:
        raise ValueError(f"Incorrect input shape, expected tuples of length {Cell.n_parameters}")
    
    if len(in_vertices.shape) == 1: #prevent errors from entering 1D arrays
        in_vertices = np.array([in_vertices])

    #2. check that the coordinate is valid in the graph lattice
    for vertex in in_vertices:

        if not(Cell.valid_coord(vertex)):
            warnings.warn(f"{vertex} is not a valid coordinate, removing from graph")
            index = get_index(vertex, in_vertices)
            in_vertices = np.delete(in_vertices, index, axis=0)

    
    in_vertices = np.unique(in_vertices, axis=0) #not sure a about axis here
    
    new_vertices = np.append(graph.vertices, in_vertices, axis=0)
    new_graph = ConfigurationGraph(Cell, new_vertices, graph.connect_type)

    #3. if necessary, check connectivity of resulting array
    if not(check_connectivity) or len(graph.vertices)==0:
        return new_graph
    else:
        if check_connectivity and not(len(graph.vertices)==0):
            if not(connected(graph)):
                warnings.warn(f'resulting graph of adding {in_vertices} would be disconnected, no vertices added')
                return graph
            else:
                return new_graph


def remove_vertices(graph: ConfigurationGraph, rm_vertices: np.array, check_connectivity=True) -> ConfigurationGraph:
    """
    Remove vertices in the graph

    parameters:
    :param graph: the graph to remove vertices from
    :param rm_vertices: the vertices to remove
    :param check_connectivity: if true, check if the resulting graph is connected

    In order for vertices to be removed, they must:
    1. be of the correct shape.
    Additionally,
    2. If the graph is required to be connected, then the reduced graph must be connected.
    """


    Cell = graph.Cell
    vertices = graph.vertices
    #1. check that the coordinates are of the correct shape
    if rm_vertices.shape[-1] != Cell.n_parameters:
        raise ValueError(f"Incorrect input shape, expected tuples of length {Cell.n_parameters}")
    
    if len(rm_vertices.shape) == 1: #prevent errors from entering 1D arrays
        rm_vertices = np.array([rm_vertices])

    
    # edit the vertices array to remove the vertices
    for vertex in rm_vertices:
        try:
            vertices = np.delete(vertices, get_index(vertex, vertices), axis=0)
        except TypeError("NoneType recieved"):
            # if the vertex is not in the graph, do nothing
            continue
    # generate new graph
    new_graph = ConfigurationGraph(Cell, vertices, graph.connect_type)


    #2. check connectivity of resulting array
    if not check_connectivity:
        return new_graph
    else:
        if not(connected(new_graph)):
            warnings.warn(f"resulting graph of removing {rm_vertices} would be disconnected, no vertices removed")
            return graph
        else:
            return new_graph


def move(Move:Move, module: np.array, direction: str, graph: ConfigurationGraph, check_connectivity=True)-> ConfigurationGraph:
    """
    Move a vertex to a new location
    
    move will be an instance of a subclass of Move, 
    if the move exists, then this will contain the transformation as a tuple
    if the move is infeasible then this move object will be None
    """
    move = Move(module, direction, graph)
    if move() is None:
        return graph
