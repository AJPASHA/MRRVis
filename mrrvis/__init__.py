"""Top-level package for mrrvis."""

__author__ = """Alexander Pasha"""
__email__ = 'alexanderjpasha@outlook.com'
__version__ = '0.1.0'


from mrrvis.cells import Square, Cube, Tri, Hex
from mrrvis.cell import Cell
from mrrvis.configuration import ConfigurationGraph, connected, equals, get_index
from mrrvis.move import Move
