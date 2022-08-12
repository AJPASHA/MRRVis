"""useful utilities"""
import numpy as np

#rotation matrices for 2D and 3D
rotation_2D = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
]).astype(int)
rotation_3Dx = lambda theta: np.array([
    [1, 0,              0], 
    [0, np.cos(theta),  -np.sin(theta)], 
    [0, np.sin(theta),  np.cos(theta)]
]).astype(int)

rotation_3Dy = lambda theta: np.array([
    [np.cos(theta), 0,  np.sin(theta)],
    [0,             1,  0],
    [-np.sin(theta), 0, np.cos(theta)]
]).astype(int)

rotation_3Dz = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta),  0],
    [0,             0,              1]
]).astype(int)