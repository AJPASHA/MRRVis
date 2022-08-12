"""Standard moveset for a square lattice"""

import numpy as np
from mrrvis.graph import ModuleGraph
from mrrvis.move import Move
from mrrvis.cells import Square
from typing import Union

class slide(Move):
    """
    Move a module within the configuration
    """
    compass = ['N', 'E', 'S', 'W']
    base_path = ['N']
    collision_mask = [False]
    def __init__(self, module_graph:ModuleGraph, module_id: Union[int,np.array], direction:str):
        super().__init__(module_graph, module_id, direction)
        self.path = self.rotate_path(direction)
        self.base_path = [self.compass.index(i) for i in self.base_path]


