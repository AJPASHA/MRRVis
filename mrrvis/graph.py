"""This module contains the module graph class"""

import numpy as np
import warnings
from mrrvis.cell import Cell
from mrrvis.cells import Square,Hex, Tri, Cube
from mrrvis.utils import rotation_2D, rotation_3Dx, rotation_3Dy, rotation_3Dz




class ModuleGraph:
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
            # print(vertices)
            self.vertices = vertices

    def get_index(self, vertex):
        """get the index of a vertex in the graph"""
        # This solution could be improved to handle multiple vertices in a single call, using a vectorised method
        try:
            return int(np.where(np.all(self.vertices == vertex, axis=1))[0][0])
        except IndexError:
            warnings.warn(f"{vertex} is not in the graph")
            return None

    def is_connected(self):
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
            [E.append(edge) for edge in self.edges_from(
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

    def isomorphic(self, other) -> bool:
        """Check if two graphs are isomorphic 
        accessible as __eq__
        currently doesn't work in 3D, hex or tri

        """
        self_verts = self.vertices
        other_verts = other.vertices

        
        #check cell types are the same and are connected in the same way
        if self.Cell != other.Cell:
            return False
            
        if self.connect_type != other.connect_type:
            return False

        rotation_angle = self.Cell.rotation_angle

        rotations_per_axis = int((2*np.pi)/rotation_angle)


        if self.Cell == Square:
            #1. center self graph on origin from most westerly, then southerly vertex
            self_min = min_coord(self_verts)
            self_verts = self_verts - self_min
            rotator = rotation_2D(rotation_angle).astype(int)
            for _ in range(rotations_per_axis):

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
                    other_verts =rotator.dot(other_verts.T).T

            return False

        elif self.Cell == Hex:
            raise NotImplementedError("isomorphic check for hex not implemented")
        elif self.Cell == Tri:
            raise NotImplementedError("isomorphic check for tri not implemented")
        elif self.Cell == Cube:
            raise NotImplementedError("isomorphic check for cube not implemented")


        
                


    def __eq__(self, other) -> bool:
        """Check if two graphs are congruent (same edge graphs)
        Currently does not work for rotational symmetry
        """
        return self.isomorphic(other)


def equals(graph1: ModuleGraph, graph2: ModuleGraph)->bool:
    """Check if two graphs are equal
    """
    return graph1 == graph2

def connected(graph: ModuleGraph):
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


        
    