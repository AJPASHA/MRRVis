
import numpy as np
from mrrvis.cells.cell import Cell

class ModuleGraph:
    def __init__(self, CellPrototype: Cell, vertices: np.array = None, connectivity='edge', check_connectivity_init=False) -> None:
        """Create a module graph object
        
        note: the default connectivity is edge, as this is the only type which is universal to all module types
        """
        #verify that Cell is a Cell Prototype
        if issubclass(CellPrototype, Cell):
            self.Cell = CellPrototype
        else:
            raise TypeError("Argument Cell must be a subclass of CellPrototype.")

        if connectivity in self.Cell.connectivity_types:
            self.connectivity = connectivity
        else:
            raise(ValueError(f"Connectivity type must be one of the following: {self.Cell.connectivity_types}"))


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
                    np.delete(vertices, index)
            # We also need to drop duplicates!
            self.vertices = vertices

    def get_index(self, vertex):
        """get the index of a vertex in the graph"""
        return int(np.where(np.all(self.vertices==vertex, axis=1))[0][0])

    
        
    def edges_from(self, vertex, connectivity=None):
        index = self.get_index

        if connectivity is None:
                connectivity = self.connectivity

        adjacents = np.array(list(self.Cell(vertex).adjacents(connectivity).values()))
        
      
        neighbors = []
        for adjacent in adjacents:
            #if there are any elements in V which are equal to adjacent
            if np.any(np.all(self.vertices==adjacent, axis=1)):
                neighbors.append({index(vertex), index(adjacent)})
        return neighbors

        # return np.array([[index(vertex), index(adjacent)] for adjacent in adjacents if adjacent in self.vertices])

    def edges_from_i(self, index, connectivity=None):
        vertex = self.vertices[index]
        return self.edges_from(vertex,connectivity)

    @property
    def edges(self):
        "returns a the edges of the graph object"
        E = []
        for vertex in self.vertices:
            [E.append(edge) for edge in self.edges_from(vertex, self.connectivity) if edge not in E]
        return E

    @property
    def V(self):
        """alias for vertices"""
        return self.vertices

    @property
    def E(self):
        "alias for edge graph"
        return self.edges

