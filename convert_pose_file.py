# converting the pose file from euler angles, translation into translation, quaternion:

import numpy as np
import math
import sys
import os
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.transforms as transforms
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import params

def get_args():
    # Parse arguments from terminal
    parser = argparse.ArgumentParser(
        description='Setting which dataset to use')
    parser.add_argument('--unity', action='store_true', default=False,
        help='Setting Unity')
    parser.add_argument('--unreal', action='store_true', default=False,
        help='Setting Unreal')
    parser.add_argument('--kitti', action='store_true', default=False,
        help='Setting KITTI')
    
    args = parser.parse_args()

    return args


def get_quaternion_from_euler(euler_angles):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll, pitch, yaw = euler_angles
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2)

    return np.array([qx, qy, qz, qw])

#REFERENCE: https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
def convert_pose_file(pose_file):
    #convert euler angles, translation into translation, quaternion
    #pose_file: pose file in euler angles, translation
    #return: pose file in translation, quaternion
    #pose_file = np.loadtxt(pose_fil
    pose_file = np.loadtxt(pose_file)
    #creating a numpy array to be filled with the new pose file:
    new_pose_arr = np.zeros((pose_file.shape[0], 7))
    print('a7a')
    print(pose_file.shape[0])

    for idx, line in enumerate(pose_file):
        translation = line[3:6]
        euler_angles = line[0:3]
        print(f'euler angles: {euler_angles}')
        print(f'translation: {translation}')
        quaternion = get_quaternion_from_euler(euler_angles)
        print(quaternion.shape)
        new_line = np.concatenate((translation, quaternion))
        # new_pose_arr.append(new_line)
        new_pose_arr[idx][:] = new_line
        print(new_line)

    return new_pose_arr



if __name__ == '__main__':
    print('Starting...')
    args = get_args()
    par = params.Parameters(args)
    pose_GT_dir = par.pose_dir  #'KITTI/pose_GT/'
    predicted_result_dir = './result/'
    gradient_color = True

    #Converting the result pose file:
    new_pose_file = convert_pose_file(predicted_result_dir + 'out_00_modified.txt')
    np.savetxt(predicted_result_dir + 'out_00_modified.txt', new_pose_file)