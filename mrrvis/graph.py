"""This module contains the module graph class"""

import numpy as np
import warnings
from mrrvis.cell import Cell




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
            vertices = np.array(vertices)
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
                    index = self.get_index(vertex)
                    vertices = np.delete(vertices, index)

            # We also need to drop duplicates!
            vertices = np.unique(vertices, axis=0)
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
        # This could potentially be reimplemented/have an alternative static option, which would be will_connect(graph1, graph2s)
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

        # return np.array([[index(vertex), index(adjacent)] for adjacent in adjacents if adjacent in self.vertices])

    def edges_from_i(self, index, connectivity=None):
        vertex = self.vertices[index]
        return self.edges_from(vertex, connectivity)

    @property
    def edges(self):
        "returns a the edges of the graph object"
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

    def __eq__(self, other) -> bool:
        raise NotImplementedError("Not implemented yet")
