# this file to add a timestamp to the pose file to convert it to TUM format:

import numpy as np
import params
import argparse

def add_timestamp(pose_file, timestamp_file):
    pose_file = np.loadtxt(pose_file)
    timestamp_file = np.loadtxt(timestamp_file)
    #Do a vstack of the two files:
    # print(new_pose_file.shape)
    print(pose_file.shape)
    #Adding a new dimension of the timestamp to match the pose file:
    timestamp_file = timestamp_file[:, np.newaxis]
    print(timestamp_file.shape)
    new_pose_file = np.hstack((timestamp_file, pose_file))
    print(new_pose_file.shape)
    return new_pose_file




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


if __name__ == '__main__':
    print('Starting...')
    args = get_args()
    par = params.Parameters(args)
    predicted_result_dir = './result/'
    pose_file = predicted_result_dir + 'out_00_modified.txt'
    timestamp_file = par.timestamp_file
    print(timestamp_file)
    new_pose_file = add_timestamp(pose_file, timestamp_file)
    np.savetxt(predicted_result_dir + 'out_00_time.txt', new_pose_file, fmt='%.6f')
    # print(f'New pose file saved to {predicted_result_dir}')
    # np.savetxt(par.new_pose_file, new_pose_file, fmt='%.6f')