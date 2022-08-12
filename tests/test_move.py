from cv2 import rotate
import pytest
import numpy as np
# from mrrvis.graph import ModuleGraph


# def test_rotate_path():
#     compass = ['N', 'E', 'S', 'W']
#     base_path = ['N', 'E', 'S']
#     assert rotate_path(compass, base_path, 'E') == ['E', 'S', 'W']
#     assert rotate_path(compass, base_path, 'S') == ['S', 'W', 'N']
#     assert rotate_path(compass, base_path, 'W') == ['W', 'N', 'E']
#     assert rotate_path(compass, base_path, 'N') == ['N', 'E', 'S']

#     assert rotate_path(compass, base_path, 'E', False) == [1,2,3]
#     assert rotate_path(compass, base_path, 'S', False) == [2,3,0]

#     base_path = [0,3,3]
#     assert rotate_path(compass, base_path, 'N') == ['N','W','W']

# def test_