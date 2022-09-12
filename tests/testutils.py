"""Utilities for testing"""

import numpy as np

def compare_np_dicts(d1:np.ndarray, d2:np.ndarray):
    """compare two dictionaries of numpy arrays"""
    return all(np.all(d1[k] == d2[k]) for k in d1.keys())