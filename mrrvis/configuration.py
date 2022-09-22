"""The configuration Module defines the ConfigurationGraph object 
and some operations which can be performed on ConfigurationGraphs

The configuration graph is a representation of a configuration in an MRR system.
It represents the collection of vertices, which are the centroids of the cells in the system,
A cell type which shows how to interpret this set of vertices as a shape,
the level of connectivity at which two cells are considered connected
An edge set constructed from the above information.

It also allows us to perform two essential operations:
    - Check if there exists an isomorphic relationship between two different configurations
    - Check to see if a configuration forms a single connected component

"""

from collections import deque
from typing import Iterable, Literal, Union
from matplotlib.pyplot import connect
import numpy as np
import warnings
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
    """A graph based representation of a reconfigurable robotics system
    
    Parameters
    ----------
    CellPrototype: Cell or {'Square', 'Cube', 'Tri', 'Hex'}
        The type of cell lattice (or alias thereof) this graph represents
    vertices: np.ndarray or array-like object, optional
        The vertices in the configuration
    connect_type: {'edge', 'vertex', 'face'}
        The level of connectivity required for two cells to be considered neighbors

    Raises
    ------
    KeyError 
        if cellPrototype is invalid
    ValueError
        if connect_type is not suitable for the given cell_type
    
    Warns
    -----
    UserWarning
        called whenever one of the vertices in the vertices array is invalid

    Attributes
    ----------
    vertices: ndarray
        the set of vertices as an array where each row is a coordinate

    
    Methods
    -------
    __eq__(other)
        (==) tests for isomorphism between this graph and another
    __contains__(other)
        (in) tests to see if the other item, a Sequence, is a vertex in self.V
    __getitem__(key)
        ConfigurationGraph()[key] obtains item from self.V by an integer index
   
    See Also
    --------
    Cell: mrrvis.cell.Cell
    """
    def __init__(self, CellPrototype: Literal['Square', 'Cube', 'Tri', 'Hex'], vertices: np.ndarray = None, connect_type='edge') -> None:
        self.Cell: Cell = _init_Cell(CellPrototype)     # identify the cell type
        self.connect_type = _init_connect_type(connect_type,self.Cell) #identify the connectivity type
        self.vertices = _init_vertices(vertices, self.Cell) #   verify the vertices

    def get_index(self, vertex: np.ndarray) -> int:
        """get the index of a vertex in the graph
        
        Parameters
        ----------
        vertex: np.ndarray
            A vertex, which should be in self.V
        
        Warns
        -----
        UserWarning
            If the vertex is not in the graph
        Returns
        -------
        int
            The index of the vertex in self.V
        """

        try:
            return int(np.where(np.all(self.vertices == vertex, axis=1))[0][0]) 
        except IndexError:
            warnings.warn(f"{vertex} is not in the graph")
            return None


    def is_reachable(self, u:int,v:int) -> bool:
        """identifies if there exists a path from the module at index u to index v
        using breadth first search
        
        Parameters
        ----------
        u: int
            The index of the first module in self.V
        v: int
            The index of the second module in self.V
        
        Returns
        -------
        bool
            True if there exists a path u ~> v
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
        return False

    def is_connected(self, connectivity:Literal['edge', 'vertex','face']=None)->bool:
        """test graph connectivity

        Performs a breadth first search for a path from the first coordinate in self.V to every
        other coordinate. is_connected==True if and only if 
        every cell is_reachable from the first cell

        Parameters
        ----------
        connectivity: {'edge', 'vertex','face'}, optional
            The level of connectivity required, by default will use this graph's declared cell_type
        
        Returns
        -------
        bool
            True if and only if the graph is connected

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
        """find the list of edges which are connected to a particular vertex
        
        Parameters
        ----------
        vertex: np.ndarray
            The coordinate to find the edges of 
        connectivity: {'edge', 'vertex','face'}, optional
            The level of connectivity at which two cell are considered connected, will default to this graph's connectivity
        omit_self: bool, default False
            Selects return type. If False will provide a list of edges represented as sets,
            if true then return just the list of the indices of the adjoining cells
        
        Returns
        -------
        list of sets of ints or list of ints
            if omit_self is true, gives a list of indices, otherwise return a list of edges represented as sets

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
        """Obtain the edges from the cell at a given index in self.V"""

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

        Parameters
        ----------
        other: ConfigurationGraph
            Another graph to compare
        
        Returns
        -------
        bool
            True if and only if self and other are isomorphic

        Notes
        -----
        if the graphs are of different connectivity types, this equation will use the type of the first operand
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
        """Add vertices to the graph (not in-place)

        Parameters
        ----------
        in_vertices: ndarray
            the vertices to add
        check_connectivity: bool, default=True
            if true, check if the resulting graph is connected the resulting graph

        Raises
        ------
        ValueError
            If in_vertices is of the wrong shape for this graph's cell type

        Warns
        -----
        UserWarning
            warning is raised if there are invalid cells in in_vertices or if the configuration is disconnected 
            and check_connectivity==True

        Returns
        -------
        ConfigurationGraph
            The edited graph or, if the connectivity check fails, self
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

    def remove_vertices(self, rm_vertices: np.ndarray, check_connectivity=True) -> 'ConfigurationGraph':
        """Remove vertices in the graph

        Parameters
        ----------
        rm_vertices: ndarray
            the vertices to remove
        check_connectivity: bool
            if true, check if the resulting graph is connected

        Raises
        ------
        ValueError
            If in_vertices is of the wrong shape for this graph's cell type

        Warns
        -----
            warning is raised if the configuration is disconnected 
            and check_connectivity==True

        Returns
        -------
        ConfigurationGraph
            the graph with the vertices removed, or, if the connectivity check fails, self
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
        """Check isomorphism of two graphs"""
        return self.isomorphic(other)

    def __getitem__(self, index):
        """access vertices as index of ConfigurationGraph"""
        return self.vertices[index]

    def __contains__(self, other: Iterable) -> bool:
        """Check whether a coordinate is in self.V"""
        return other.tolist() in self.vertices.tolist()


def vert_connected(graph: ConfigurationGraph) -> bool:
    """Checks if the graph is vertex connected
    
    See Also
    --------
    is_connected: ConfigurationGraph.is_connected

    Notes
    -----
        this is used for checking moves
    """
    return graph.is_connected('vertex')


def edge_connected(graph: ConfigurationGraph) -> bool:
    """checks if the graph is edge connected
    
    See Also
    --------
    is_connected: ConfigurationGraph.is_connected

    Notes
    -----
    this is used for checking moves
    """
    return graph.is_connected('edge')


def get_index(vertex: np.array, vertices: np.ndarray) -> int:
    """get the index of a vertex in a set of vertices

    See Also
    --------
    get_index: graph.get_index
    """
    # This solution could be improved to handle multiple vertices in a single call, using a vectorised method
    try:
        return int(np.where(np.all(vertices == vertex, axis=1))[0][0])
    except IndexError:
        return None


def min_coord(vertices: np.ndarray) -> np.ndarray:
    """find the canonical minimum coordinate in a set of vertices

    Parameters
    ----------
    vertices: ndarray

    Returns
    -------
    ndarray
        The minimum coordinate in the input array

    Notes
    -----
    This minimum works by finding the smallest x, smallest y and then smallest z in vertices
    """
    # we perform this iteratively to deal with the existence of multiple minimum coordinates
    # depending on the rotation and shape
    # we can make use of the fact that to add a vector to the matrix, it must be unique
    for i in range(vertices.shape[-1]):
        min_index = np.argmin(vertices[:, i])

        if type(min_index) == np.int64:

            return vertices[min_index]


def max_coord(vertices: np.ndarray) -> np.ndarray:
    """find the maximum coordinate in a set of vertices

    Parameters
    ----------
    vertices: ndarray

    Returns
    -------
    ndarray
        The maximum coordinate in the input array

    Notes
    -----
    This maximum works by finding the largest x, largest y and then largest z in vertices
    """
    # we perform this iteratively to deal with the existence of multiple minimum coordinates
    # depending on the rotation and shape
    # we can make use of the fact that to add a vector to the matrix, it must be unique
    for i in range(vertices.shape[-1]):
        max_index = np.argmax(vertices[:, i])

        if type(max_index) == np.int64:

            return vertices[max_index]


def add_vertices(graph: ConfigurationGraph, in_vertices: np.ndarray, check_connectivity=True) -> ConfigurationGraph:
    """Add vertices to the graph

    See Also
    --------
    add_vertices: ConfigurationGraph.add_vertices
    """
    return graph.add_vertices(in_vertices, check_connectivity)


def remove_vertices(graph: ConfigurationGraph, rm_vertices: np.ndarray, check_connectivity=True) -> ConfigurationGraph:
    """
    Remove vertices in the graph

    See Also
    --------
    rm_vertices: ConfigurationGraph.rm_vertices
    """
    return graph.remove_vertices(rm_vertices, check_connectivity)
