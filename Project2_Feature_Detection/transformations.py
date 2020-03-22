import math
import numpy as np


def get_rot_mx(angle_x, angle_y, angle_z):
    '''
    Input:
        angle_x -- Rotation around the x axis in radians
        angle_y -- Rotation around the y axis in radians
        angle_z -- Rotation around the z axis in radians
    Output:
        A 4x4 numpy array representing 3D rotations. The order of the rotation
        axes from first to last is x, y, z, if you multiply with the resulting
        rotation matrix from left.
    '''
    # Note: For MOPS, you need to use angle_z only, since we are in 2D

    rot_x_mx = np.array([[1, 0, 0, 0],
                         [0, math.cos(angle_x), -math.sin(angle_x), 0],
                         [0, math.sin(angle_x), math.cos(angle_x), 0],
                         [0, 0, 0, 1]])

    rot_y_mx = np.array([[math.cos(angle_y), 0, math.sin(angle_y), 0],
                         [0, 1, 0, 0],
                         [-math.sin(angle_y), 0, math.cos(angle_y), 0],
                         [0, 0, 0, 1]])

    rot_z_mx = np.array([[math.cos(angle_z), -math.sin(angle_z), 0, 0],
                         [math.sin(angle_z), math.cos(angle_z), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    return np.dot(rot_z_mx, np.dot(rot_y_mx, rot_x_mx))


def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 3
        elements
    Output:
        A 4x4 numpy array representing 3D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 3

    trans_mx = np.eye(4)
    trans_mx[:3, 3] = trans_vec

    return trans_mx


def get_scale_mx(s_x, s_y, s_z):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
        s_z -- Scaling along the z axis
    Output:
        A 4x4 numpy array representing 3D scaling.
    '''
    # Note: For MOPS, you need to use s_x and s_y only, since we are in 2D
    scale_mx = np.eye(4)

    for i, s in enumerate([s_x, s_y, s_z]):
        scale_mx[i, i] = s

    return scale_mx

