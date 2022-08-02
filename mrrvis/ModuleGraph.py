print(__package__)
import numpy as np
from .Cell import Cell
from .Square import Square

class ModuleGraph:
    def __init__(self, Cell: Cell, vertices: np.array = np.array([]), connectivity='edge', check_connectivity_init=False) -> None:
        
        #verify that Cell is a Cell Prototype
        if issubclass(Cell, Cell):
            self.Cell = Cell
        else:
            raise TypeError("Argument Cell must be a subclass of CellPrototype.")
        #note for testing, we will need to check the case of what happens if the abstract class is wrongly invoked
        #I think that ABC should through a NotImplementedError, but we need to check


        if len(vertices) != 0:  # check that vertices are of the correct shape
            if vertices.shape[-1] != self.Cell.n_parameters:
                raise ValueError(
                    """module array rows must be of same length as Cell.n_parameters"""
                    )

        for vertex in vertices:  # Verify that the modules in the modules matrix are valid coordinates, if not issue warning and drop those cells
            if not(Cell.valid_coord(vertex)):
                print(
                    f"invalid cell detected at {vertex}, dropping from graph")
                index = np.where(np.all(vertices == vertex, axis=1))
                np.delete(vertices, index)

        self.vertices = vertices

        self.connectivity = connectivity

    def edges_from(self, vertex, connectivity=None):

        if connectivity == None:
            connectivity = self.connectivity

        adjacents = self.Cell(vertex).adjacents(connectivity)

        for adjacent in adjacents:
            if adjacent in self.vertices:
                yield (vertex, adjacent)

    @property
    def edges(self):
        "returns a the edges of the graph object"
        E = []
        for vertex in self.vertices:
            E.append(self.edges_from(vertex, self.connectivity))

    @property
    def V(self):
        """alias for vertices"""
        return self.vertices

    @property
    def E(self):
        "alias for edge graph"
        return self.edges


def main():
    vertices = np.array([
        [1, 1],
        [5, 0],
        [3, 4],

    ])
    graph = ModuleGraph(Square, vertices)


if __name__ == '__main__':
    main()
