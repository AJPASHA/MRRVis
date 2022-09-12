"""This module contains the module graph class"""

from collections import deque
from typing import Iterable, Literal, Union
from matplotlib.pyplot import connect
import numpy as np
import warnings

from sqlalchemy import false
from mrrvis.cell import Cell, Square, Hex, Tri, Cube
from mrrvis.geometry_utils import rotate_normal, cube_rotation_list

cells = {
    "Square": Square,
    "Cube": Cube,
    "Tri": Tri,
    "Hex": Hex,
}

# These private methods help to tidy up a lot of the initialisation logic, because there are many checks that we want to make on a new graph,
# so it helps to compartmentalise them a bit

def _init_Cell(cell_input: Union[str, Cell]):
    """Private. define the cell type from either a string or direct input
    Used for initialising ConfigurationGraphs
    """
    if type(cell_input) is not str:
        # if the initialiser has taken a cell class just return that
        if issubclass(cell_input, Cell):
            return cell_input

    try:
        return cells[cell_input.capitalize()]
    except KeyError as e:
        raise KeyError(
            "Cell type not found, must be one of 'Square', 'Cube', 'Tri', 'Hex'")

def _init_connect_type(connect_type: str, Cell: Cell):
    """Private. identify the connectivity type based on the arguments
    used for initialising ConfigurationGraphs"""
    if connect_type not in Cell.connectivity_types:
        raise(ValueError(
                f"Connectivity type must be one of the following: {Cell.connectivity_types}"))
    return connect_type


def _init_vertices(vertices: Union[np.ndarray, None], Cell: Cell):
    """Private. initialise vertices
    Used to initialise ConfigurationGraphs
    """
    if vertices is None:
        # provide an empty array if None is provided
        return np.empty((0, Cell.n_parameters))

    # cast, in case a list or something gets passed in
    vertices = np.array(vertices)

    if len(vertices.shape) == 1:
        vertices = np.array([vertices])                     # Make array 2D

    # check that there are the right number of columns (parameters) for the cell type
    if vertices.shape[-1] != Cell.n_parameters:
        raise ValueError(
            """module array rows must be of same length as Cell.n_parameters"""
        )

    for vertex in vertices:                                 # remove invalid vertices, issuing warnings for doing so
        if not(Cell.valid_coord(vertex)):
            warnings.warn(
                f"invalid cell detected at {vertex}, dropping from graph")
            vertices = np.delete(
                vertices, get_index(vertex, vertices), axis=0)

    vertices, ind = np.unique(
        vertices, return_index=True, axis=0)  # remove duplicates
    vertices = vertices[np.argsort(ind)]

    return vertices


class ConfigurationGraph:
    def __init__(self, CellPrototype: Literal['Square', 'Cube', 'Tri', 'Hex'], vertices: np.ndarray = None, connect_type='edge') -> None:
        """Create a module graph object

        :param CellPrototype: the cell class to use for the graph
        :type CellPrototype: Literal['Square', 'Cube', 'Tri', 'Hex'] or mrrvis.Cell subclass
        :param vertices: the vertices of the graph, if None, an empty set of vertices will be generated
        :type vertices: np.ndarray or list of lists
        :param connect_type: the type of connectivity to use for the graph, can be 'face', 'edge' or 'vertex' 
                depending on the type of cell the graph is based on*

        * to check this for a Cell, use the Cell.connectivity_types property

        note: the default connectivity is edge, as this is the only type which is universal to all module types
        """

        self.Cell: Cell = _init_Cell(CellPrototype)
        self.connect_type = _init_connect_type(connect_type,self.Cell)
        self.vertices = _init_vertices(vertices, self.Cell)

    def get_index(self, vertex: np.ndarray) -> int:
        """get the index of a vertex in the graph
        :param vertex: the vertex coordinate in the graph
        :return: an integer index of the coordinate in self.V
        """

        try:
            return int(np.where(np.all(self.vertices == vertex, axis=1))[0][0]) 
        except IndexError:
            warnings.warn(f"{vertex} is not in the graph")
            return None


    def is_reachable(self, u:int,v:int) -> bool:
        """identifies if there exists a path from the module at index u to index v
        using breadth first search
        :param u: index of the first cell
        :param v: index of the other cell
        :return: bool representing whether or not a path exists
        used for checking connectivity in graphs
        """
        if u==v:
            return True

        visited = [False for i in range(len(self.V))]
        queue = deque()

        visited[u] = True
        queue.append(u)

        while len(queue)>0:
            u = queue.popleft()
            for i in self.edges_from_i(u,omit_self=True):
                if i==v:
                    return True
                if not visited[i]:
                    visited[i] = True
                    queue.append(i)

    def is_connected(self, connectivity:Literal['edge', 'vertex','face']=None)->bool:
        """test graph connectivity.
        :param connectivity: the level of connectivity required, depending on self.Cell, could be any of 'edge', 'vertex', 'face' 
        :return: bool representing the status of graph connectivity
        This is done by performing a breadth first search of every module from the zeroth module to see
        if every module is reachable from the zeroth module, if so, then the graph is connected
        """
        if connectivity is None:
            connectivity=self.connect_type

        if len(self.V)<=1:
            return True

        for i in range(len(self.V)):
            if not self.is_reachable(0, i):
                return False
        return True
        


    def edges_from(self, vertex: np.ndarray, connectivity:Literal['edge', 'vertex','face']=None, omit_self=False):
        """edges which are connected to a particular vertex
        :param vertex: the coordinate to find edges from
        :param connectivity: the connectivity level required of neighbors
        :param omit_self: returns a list of edges if false, or the indices of connecting vertices if true
        :return: a list of undirected edges if omit_self if False
        :return: a list of connecting node indices if omit_self
        :rtype: List[set] or List[int]
        """
        index = self.get_index

        if connectivity is None:
            connectivity = self.connect_type

        adjacents = np.array(
            list(self.Cell(vertex).adjacents(connectivity).values()))
  
        neighbors = []
        for adjacent in adjacents:
            # if there are any elements in V which are equal to adjacent
            if np.any(np.all(self.vertices == adjacent, axis=1)):
                if not omit_self:
                    neighbors.append({index(vertex), index(adjacent)})
                else:
                    neighbors.append(index(adjacent))

        return neighbors

    def edges_from_i(self, index, connectivity=None, omit_self=False):

        vertex = self.vertices[index]
        return self.edges_from(vertex, connectivity, omit_self=omit_self)

    @property
    def edges(self):
        """returns the edges of the graph object"""
        E = []
        for vertex in self.vertices:
            _ = [E.append(edge) for edge in self.edges_from(
                vertex, self.connect_type) if edge not in E]
        return E

    @property
    def V(self):
        """the vertices of the graph"""
        return self.vertices

    @property
    def E(self):
        "the edges of the graph"
        return self.edges

    @property
    def n(self):
        "number of modules in the graph"
        return len(self.vertices)

    def isomorphic(self, other: 'ConfigurationGraph') -> bool:
        """Check if two graphs are isomorphic 

        :param other: the graph to compare
        :return: bool representing whether the two graphs are isomorphic or not

        accessible as ConfigurationGraph.__eq__

        note, if the graphs are of different connectivity types, this equation will use the type of the first operand

        note, this is currently a brute force method, which is fast enough for two dimensional cell types
        who's rotation groups have an cardinality equal to the number of vertices in the shape, but slow for cubes.
        This is because the order of the rotation group of a shape on a 3D discrete lattice is 24, which is still small enough to be brute forced,
        but in future a more efficient method might need to be considered if possible
        """

        self_verts = self.vertices
        other_verts = other.vertices


        if self.Cell != other.Cell:         # graphs can only be isomorphic if cell types are homogenous
            return False


        self_min = min_coord(self_verts)    # identify a canonical 'minimum coordninate' according to smallest x, then y, then z

        self_verts = self_verts - self_min  # translate self shape to origin

        if self.Cell.dimensions == 2:

            num_rots = (2*np.pi)/self.Cell.rotation_angle
            if abs(num_rots-int(num_rots)) > 0.001:
                raise ValueError(
                    "rotation angle must be a multiple of 2*pi/n_parameters")
            num_rots = int(num_rots)

            for _ in range(num_rots):

                # 2. identify the 'smallest' vertex in other graph and translate to origin
                other_min = min_coord(other_verts)
                other_verts_temp = other_verts - other_min
                # 3. verify that the graphs are isomorphic based on their vertex sets
                self_set = set([tuple(vert) for vert in self_verts])
                other_set = set([tuple(vert) for vert in other_verts_temp])
                if self_set == other_set:
                    return True
                # 4. rotate the other graph by the rotation angle
                else:
                    if self.Cell == Square:
                        other_verts = rotate_normal(other_verts, 1)
                    elif self.Cell in (Hex, Tri):
                        # not sure about this, will need further testing
                        other_verts = rotate_normal(
                            other_verts, 2, axis=np.array([1, 1, 1]))

        if self.Cell.dimensions == 3:
            # will need to test this further
            other_min = min_coord(other_verts)
            other_verts = other_verts - other_min
            rotation_generator = cube_rotation_list(other_verts)
            for other_verts in rotation_generator:
                self_set = set([tuple(vert) for vert in self_verts])
                other_set = set([tuple(vert) for vert in other_verts])
                if self_set == other_set:
                    return True
        return False

    def add_vertices(self, in_vertices: np.array, check_connectivity=True) -> 'ConfigurationGraph':
        """
        Add vertices to the graph

        :param in_vertices: the vertices to add
        :param check_connectivity: if true, check if the resulting graph is connected the resulting graph
        :return: configuration graph with additional vertices
        :raise ValueError: if in_vertices are of wrong shape


        In order for vertices to be valid, they must:
        1. be of the correct shape,
        2. be valid coordinates.
        3. if the graph is required to be connected, then the vertices must form a connected graph when concatenated with existing vertices.
        """
        Cell = self.Cell
        in_vertices = np.array(in_vertices)

        # 1. check that the coordinates are of the correct shape
        if in_vertices.shape[-1] != Cell.n_parameters:
            raise ValueError(
                f"Incorrect input shape, expected tuples of length {Cell.n_parameters}")

        if len(in_vertices.shape) == 1:  # prevent errors from entering 1D arrays
            in_vertices = np.array([in_vertices])

        # 2. check that the coordinate is valid in the graph lattice
        for vertex in in_vertices:

            if not(Cell.valid_coord(vertex)):
                warnings.warn(
                    f"{vertex} is not a valid coordinate, removing from graph")
                index = get_index(vertex, in_vertices)
                in_vertices = np.delete(in_vertices, index, axis=0)

        # not sure a about axis here
        in_vertices = np.unique(in_vertices, axis=0)

        new_vertices = np.append(self.vertices, in_vertices, axis=0)
        new_graph = ConfigurationGraph(Cell, new_vertices, self.connect_type)

        # 3. if necessary, check connectivity of resulting array
        if not(check_connectivity) or len(self.vertices) == 0:
            return new_graph
        else:
            if check_connectivity and not(len(self.vertices) == 0):
                if not(self.is_connected()):
                    warnings.warn(
                        f'resulting graph of adding {in_vertices} would be disconnected, no vertices added')
                    return self
                else:
                    return new_graph

    def remove_vertices(self, rm_vertices: np.array, check_connectivity=True) -> 'ConfigurationGraph':
        """
        Remove vertices in the graph


        :param rm_vertices: the vertices to remove
        :param check_connectivity: if true, check if the resulting graph is connected
        :raise ValueError: if in_vertices are of wrong shape
        :return: the graph with the vertices removed

        In order for vertices to be removed, they must:
        1. be of the correct shape.
        Additionally,
        2. If the graph is required to be connected, then the reduced graph must be connected.
        """

        Cell = self.Cell
        vertices = self.vertices
        # check that the coordinates are of the correct shape
        if rm_vertices.shape[-1] != Cell.n_parameters:
            raise ValueError(
                f"Incorrect input shape, expected tuples of length {Cell.n_parameters}")

        if len(rm_vertices.shape) == 1:  # prevent errors from entering 1D arrays
            rm_vertices = np.array([rm_vertices])

        # edit the vertices array to remove the vertices
        indices = [self.get_index(vertex) for vertex in rm_vertices]
        vertices = np.delete(vertices, indices, axis=0)

        new_graph = ConfigurationGraph(Cell, vertices, self.connect_type)
        # 2. check connectivity of resulting array
        if not check_connectivity:

            return new_graph
        else:
            if not(new_graph).is_connected():
                warnings.warn(
                    f"resulting graph of removing {rm_vertices} would be disconnected, no vertices removed")
                return self
            else:

                return new_graph

    def __eq__(self, other) -> bool:
        """Check if two graphs are congruent (same edge graphs)
        Currently does not work for rotational symmetry
        """
        return self.isomorphic(other)

    def __getitem__(self, index):
        return self.vertices[index]

    def __contains__(self, other: Iterable) -> bool:
        return other.tolist() in self.vertices.tolist()


def equals(graph1: ConfigurationGraph, graph2: ConfigurationGraph) -> bool:
    """Check if two graphs are equal
    """
    return graph1 == graph2


def vert_connected(graph: ConfigurationGraph) -> bool:
    """Checks if the graph is vertex connected
    alias of graph.is_connected('vertex')
    """
    return graph.is_connected('vertex')


def edge_connected(graph: ConfigurationGraph) -> bool:
    """checks if the graph is edge connected
    alias of graph.is_connected('edge')
    """
    return graph.is_connected('edge')


def get_index(vertex: np.array, vertices: np.array) -> int:
    """get the index of a vertex in a set of vertices
    """
    # This solution could be improved to handle multiple vertices in a single call, using a vectorised method
    try:
        return int(np.where(np.all(vertices == vertex, axis=1))[0][0])
    except IndexError:
        return None


def min_coord(vertices: np.array) -> np.array:
    """find the canonical minimum coordinate in a set of vertices

    This minimum works by finding the smallest x, smallest y and then smallest z in vertices
    """
    # we perform this iteratively to deal with the existence of multiple minimum coordinates
    # depending on the rotation and shape
    # we can make use of the fact that to add a vector to the matrix, it must be unique
    for i in range(vertices.shape[-1]):
        min_index = np.argmin(vertices[:, i])

        if type(min_index) == np.int64:

            return vertices[min_index]


def max_coord(vertices: np.array) -> np.array:
    """find the maximum coordinate in a set of vertices

    parameters:
    :param vertices: the set of vertices to search
    :return: the index of the vertex in the set of vertices
    """
    # we perform this iteratively to deal with the existence of multiple minimum coordinates
    # depending on the rotation and shape
    # we can make use of the fact that to add a vector to the matrix, it must be unique
    for i in range(vertices.shape[-1]):
        max_index = np.argmax(vertices[:, i])

        if type(max_index) == np.int64:

            return vertices[max_index]


def add_vertices(graph: ConfigurationGraph, in_vertices: np.array, check_connectivity=True) -> ConfigurationGraph:
    """Add vertices to the graph

    :param graph: the graph to edit
    :param in_vertices: the vertices to add
    :param check_connectivity: if true, check if the resulting graph is connected the resulting graph
    :return: configuration graph with additional vertices
    :raise ValueError: if in_vertices are of wrong shape

    In order for vertices to be valid, they must:
    1. be of the correct shape,
    2. be valid coordinates.
    3. if the graph is required to be connected, then the vertices must form a connected graph when concatenated with existing vertices.
    """
    return graph.add_vertices(in_vertices, check_connectivity)


def remove_vertices(graph: ConfigurationGraph, rm_vertices: np.array, check_connectivity=True) -> ConfigurationGraph:
    """
    Remove vertices in the graph

    :param graph: the graph to edit
    :param rm_vertices: the vertices to remove
    :param check_connectivity: if true, check if the resulting graph is connected
    :raise ValueError: if in_vertices are of wrong shape
    :return: the graph with the vertices removed

    In order for vertices to be removed, they must:
    1. be of the correct shape.
    Additionally,
    2. If the graph is required to be connected, then the reduced graph must be connected.
    """
    return graph.remove_vertices(rm_vertices, check_connectivity)
