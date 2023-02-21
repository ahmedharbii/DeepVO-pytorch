import numpy as np



path_kitti = '/home/mrblack/Projects_DL/DeepVO-pytorch/KITTI/pose_GT/00.txt'

path_unity = '/home/mrblack/Projects_DL/DeepVO-pytorch/Unity/back_forward/pose_left_kitti/00.txt'



# Compare KITTI and Unity

kitti = np.loadtxt(path_kitti)
unity = np.loadtxt(path_unity)

print(kitti.shape)
print(unity.shape)

print(kitti.size)
print(unity.size)


