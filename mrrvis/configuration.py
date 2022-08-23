"""This module contains the module graph class"""

from typing import Iterable
import numpy as np
import warnings
from mrrvis.cell import Cell, Square,Hex, Tri, Cube
from mrrvis.geometry_utils import rotate_normal, cube_rotation_list




class ConfigurationGraph:
    def __init__(self, CellPrototype: Cell, vertices: np.array = None, connect_type='edge') -> None:
        """Create a module graph object
        parameters:
        :param CellPrototype: the cell class to use for the graph
        :param vertices: the vertices of the graph, if None, an empty set of vertices will be generated
        :param connect_type: the type of connectivity to use for the graph, can be 'face', 'edge' or 'vertex' 
                depending on the type of cell the graph is based on*

        * to check this for a Cell, use the Cell.connectivity_types property

        note: the default connectivity is edge, as this is the only type which is universal to all module types
        """
        # verify that Cell is a Cell Prototype
        if issubclass(CellPrototype, Cell):
            self.Cell = CellPrototype
        else:
            raise TypeError(
                "Argument Cell must be a subclass of CellPrototype.")

        if connect_type in self.Cell.connectivity_types:
            self.connect_type = connect_type
        else:
            raise(ValueError(
                f"Connectivity type must be one of the following: {self.Cell.connectivity_types}"))

        # This is messy, needs refactoring, but it works for now
        if vertices is None:
            self.vertices = np.empty((0, self.Cell.n_parameters))
        else:
            if len(vertices.shape) ==1:
                vertices = np.array([vertices])
            vertices = np.array(vertices).astype(int)

            # check that vertices are of the correct shape

            if vertices.shape[-1] != self.Cell.n_parameters:
                raise ValueError(
                    """module array rows must be of same length as Cell.n_parameters"""
                )
            # Verify that the modules in the modules matrix are valid coordinates, if not issue warning and drop those cells
            for vertex in vertices:
                if not(self.Cell.valid_coord(vertex)):
                    print(
                        f"invalid cell detected at {vertex}, dropping from graph")

                    #this doesn't make sense?
                    
                    vertices = np.delete(vertices, get_index(vertex, vertices), axis=0)
            
            # We also need to drop duplicates!
            # np.unique doesn't preserve order, so we need to record the indices and perform argsort on the resulting array
            vertices, ind = np.unique(vertices, return_index=True,axis=0)
            vertices =  vertices[np.argsort(ind)]
  
            self.vertices = vertices

    def get_index(self, vertex):
        """get the index of a vertex in the graph"""
        # This solution could be improved to handle multiple vertices in a single call, using a vectorised method
        try:
            return int(np.where(np.all(self.vertices == vertex, axis=1))[0][0])
        except IndexError:
            warnings.warn(f"{vertex} is not in the graph")
            return None

    def is_connected(self) -> bool:
        """Check if the graph is connected"""
        vertices = self.vertices

        # if there are fewer than 2 vertices, the graph is connected
        if len(vertices)<=1:
            return True

        for vertex in vertices:
            if len(self.edges_from(vertex)) == 0:
                warnings.warn(f"{vertex} is not connected to the graph")
                return False

        return True

    def edges_from(self, vertex, connectivity=None):
        index = self.get_index

        if connectivity is None:
            connectivity = self.connect_type

        adjacents = np.array(
            list(self.Cell(vertex).adjacents(connectivity).values()))

        neighbors = []
        for adjacent in adjacents:
            # if there are any elements in V which are equal to adjacent
            if np.any(np.all(self.vertices == adjacent, axis=1)):
                neighbors.append({index(vertex), index(adjacent)})
        return neighbors

    def edges_from_i(self, index, connectivity=None):

        vertex = self.vertices[index]
        return self.edges_from(vertex, connectivity)

    @property
    def edges(self):
        "returns the edges of the graph object"
        E = []
        for vertex in self.vertices:
            _ = [E.append(edge) for edge in self.edges_from(
                vertex, self.connect_type) if edge not in E]
        return E

    @property
    def V(self):
        """alias for vertices"""
        return self.vertices

    @property
    def E(self):
        "alias for edge graph"
        return self.edges

    @property
    def n(self):
        "number of modules in the graph"
        return len(self.vertices)

    def isomorphic(self, other) -> bool:
        """Check if two graphs are isomorphic 
        accessible as ConfigurationGraph.__eq__
        currently doesn't work in 3D
        note, if the graphs are of different connectivity types, this equation will use the type of the first operand

        note, this is currently a brute force method, which is fast enough for two dimensional cell types
        who's rotation groups have an cardinality equal to the number of vertices in the shape, but slow for cubes.
        This is because the order of the rotation group of a cube is 24, which is still small enough to be brute forced,
        but in future a more efficient method might need to be considered
        """
  
        self_verts = self.vertices
        other_verts = other.vertices

        
        #check cell types are the same 
        if self.Cell != other.Cell:
            return False
            
        #1. center self graph on origin from most westerly, then southerly, then downward vertex
        self_min = min_coord(self_verts)
        self_verts = self_verts - self_min

        if self.Cell.dimensions==2:

            num_rots = (2*np.pi)/self.Cell.rotation_angle
            if abs(num_rots-int(num_rots))>0.001:
                raise ValueError("rotation angle must be a multiple of 2*pi/n_parameters")
            num_rots = int(num_rots)

            for _ in range(num_rots):

                #2. identify the 'smallest' vertex in other graph and translate to origin
                other_min = min_coord(other_verts)
                other_verts_temp = other_verts - other_min
                #3. verify that the graphs are isomorphic based on their vertex sets
                self_set = set([tuple(vert) for vert in self_verts])
                other_set = set([tuple(vert) for vert in other_verts_temp])
                if self_set == other_set:
                    return True
                #4. rotate the other graph by the rotation angle
                else:
                    if self.Cell == Square:
                        other_verts = rotate_normal(other_verts, 1)
                    elif self.Cell in (Hex, Tri):
                        #not sure about this, will need further testing
                        other_verts = rotate_normal(other_verts,2, axis=np.array([1,1,1]))

        if self.Cell.dimensions==3:
            # will need to test this further
            other_min = min_coord(other_verts)
            other_verts = other_verts - other_min
            rotation_generator = cube_rotation_list(other_verts)
            for other_verts in rotation_generator:
                self_set = set([tuple(vert) for vert in self_verts])
                other_set = set([tuple(vert) for vert in other_verts])
                if self_set == other_set:
                    return True

    def add_vertices(self, in_vertices: np.array, check_connectivity=True) -> 'ConfigurationGraph':
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
        Cell = self.Cell
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
        
        new_vertices = np.append(self.vertices, in_vertices, axis=0)
        new_graph = ConfigurationGraph(Cell, new_vertices, self.connect_type)

        #3. if necessary, check connectivity of resulting array
        if not(check_connectivity) or len(self.vertices)==0:
            return new_graph
        else:
            if check_connectivity and not(len(self.vertices)==0):
                if not(self.is_connected()):
                    warnings.warn(f'resulting graph of adding {in_vertices} would be disconnected, no vertices added')
                    return self
                else:
                    return new_graph

    def remove_vertices(self, rm_vertices: np.array, check_connectivity=True) -> 'ConfigurationGraph':
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


        Cell = self.Cell
        vertices = self.vertices
        #1. check that the coordinates are of the correct shape
        if rm_vertices.shape[-1] != Cell.n_parameters:
            raise ValueError(f"Incorrect input shape, expected tuples of length {Cell.n_parameters}")
        
        if len(rm_vertices.shape) == 1: #prevent errors from entering 1D arrays
            rm_vertices = np.array([rm_vertices])

        
        # edit the vertices array to remove the vertices
        for vertex in rm_vertices:
            try:
                vertices = np.delete(vertices, self.get_index(vertex), axis=0)
            except TypeError("None Type recieved"):
                # if the vertex is not in the graph, do nothing
                continue
        # generate new graph
        new_graph = ConfigurationGraph(Cell, vertices, self.connect_type)


        #2. check connectivity of resulting array
        if not check_connectivity:
            return new_graph
        else:
            if not(new_graph).is_connected():
                warnings.warn(f"resulting graph of removing {rm_vertices} would be disconnected, no vertices removed")
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

    def __contains__ (self, other: Iterable)->bool:
        return other.tolist() in self.vertices.tolist()


def equals(graph1: ConfigurationGraph, graph2: ConfigurationGraph)->bool:
    """Check if two graphs are equal
    """
    return graph1 == graph2

def connected(graph: ConfigurationGraph):
    """Checks if the graph is connected
    alias of graph.is_connected()
    """
    return graph.is_connected()

def get_index(vertex:np.array, vertices:np.array)->int:
        """get the index of a vertex in a set of vertices
        
        parameters:
        :param vertex: the vertex to find the index of
        :param vertices: the set of vertices to search
        :return: the index of the vertex in the set of vertices
        """
        # This solution could be improved to handle multiple vertices in a single call, using a vectorised method
        try:
            return int(np.where(np.all(vertices == vertex, axis=1))[0][0])
        except IndexError:
            return None

def min_coord(vertices:np.array)->np.array:
    """find the minimum coordinate in a set of vertices
    
    parameters:
    :param vertices: the set of vertices to search
    :return: the index of the vertex in the set of vertices
    """
    #we perform this iteratively to deal with the existence of multiple minimum coordinates 
    # depending on the rotation and shape
    # we can make use of the fact that to add a vector to the matrix, it must be unique
    for i in range(vertices.shape[-1]):
        min_index = np.argmin(vertices[:,i])

        if type(min_index) == np.int64:

            return vertices[min_index]

def max_coord(vertices:np.array)->np.array:
    """find the maximum coordinate in a set of vertices
    
    parameters:
    :param vertices: the set of vertices to search
    :return: the index of the vertex in the set of vertices
    """
    #we perform this iteratively to deal with the existence of multiple minimum coordinates 
    # depending on the rotation and shape
    # we can make use of the fact that to add a vector to the matrix, it must be unique
    for i in range(vertices.shape[-1]):
        max_index = np.argmax(vertices[:,i])

        if type(max_index) == np.int64:

            return vertices[max_index]


def add_vertices(graph: ConfigurationGraph, in_vertices: np.array, check_connectivity=True) -> ConfigurationGraph:
    return graph.add_vertices(in_vertices, check_connectivity)

def remove_vertices(graph: ConfigurationGraph, rm_vertices: np.array, check_connectivity=True) -> ConfigurationGraph:
    return graph.remove_vertices(rm_vertices, check_connectivity)
