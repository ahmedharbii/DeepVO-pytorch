import os
import glob
import numpy as np
import time
from helper import R_to_angle
# from params import par
import params
from torchvision import transforms
from PIL import Image
import torch
import math
import sys
import argparse
# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data():
	# info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
	# info = {'00' : [0,1293]}
	info = {'00' : [0, 4765]}
	# print('Creating pose data...')
	# print(info.keys())
	start_t = time.time()
	for video in info.keys():
		fn = '{}{}.txt'.format(par.pose_dir, video)
		print('Transforming {}...'.format(fn))
		with open(fn) as f:
			lines = [line.split('\n')[0] for line in f.readlines()] 
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]  # list of pose (pose=list of 12 floats)
			poses = np.array(poses)
			base_fn = os.path.splitext(fn)[0]
			np.save(base_fn+'.npy', poses)
			print('Video {}: shape={}'.format(video, poses.shape))
	print('elapsed time = {}'.format(time.time()-start_t))



def calculate_rgb_mean_std(image_path_list, minus_point_5=True):
	n_images = len(image_path_list)
	# print(image_path_list)
	cnt_pixels = 0
	print('Numbers of frames in training dataset: {}'.format(n_images))
	mean_np = [0, 0, 0]
	mean_tensor = [0, 0, 0]
	
	to_tensor = transforms.ToTensor()
	# print('img_path_list', image_path_list)
	for idx, img_path in enumerate(image_path_list):
		print('{} / {}'.format(idx, n_images), end='\r')
		# print('img_path:', img_path)
		img_as_img = Image.open(img_path)
		# print(img_as_img.shape)
		img_as_tensor = to_tensor(img_as_img)
		print(img_as_tensor.shape)
		if minus_point_5:
			img_as_tensor = img_as_tensor - 0.5
		img_as_np = np.array(img_as_img)
		img_as_np = np.rollaxis(img_as_np, 2, 0)
		print(img_as_np.shape)
		cnt_pixels += img_as_np.shape[1]*img_as_np.shape[2]
		for c in range(3):
			mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
			mean_np[c] += float(np.sum(img_as_np[c]))
	mean_tensor =  [v / cnt_pixels for v in mean_tensor]
	mean_np = [v / cnt_pixels for v in mean_np]
	print('mean_tensor = ', mean_tensor)
	print('mean_np = ', mean_np)

	std_tensor = [0, 0, 0]
	std_np = [0, 0, 0]
	for idx, img_path in enumerate(image_path_list):
		print('{} / {}'.format(idx, n_images), end='\r')
		img_as_img = Image.open(img_path)
		img_as_tensor = to_tensor(img_as_img)
		if minus_point_5:
			img_as_tensor = img_as_tensor - 0.5
		img_as_np = np.array(img_as_img)
		img_as_np = np.rollaxis(img_as_np, 2, 0)
		for c in range(3):
			tmp = (img_as_tensor[c] - mean_tensor[c])**2
			std_tensor[c] += float(torch.sum(tmp))
			tmp = (img_as_np[c] - mean_np[c])**2
			std_np[c] += float(np.sum(tmp))
	std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
	std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
	print('std_tensor = ', std_tensor)
	print('std_np = ', std_np)

def get_args():
	# Parse arguments from terminal
	parser = argparse.ArgumentParser(
		description='Setting which dataset to use')
	parser.add_argument('--unity', action='store_true', default=False,
		help='Setting Unity')
	parser.add_argument('--unreal', action='store_true', default=True,
		help='Setting Unreal')
	parser.add_argument('--kitti', action='store_true', default=False,
		help='Setting KITTI')
	
	args = parser.parse_args()

	return args

if __name__ == '__main__':

	args = get_args()
	print(args)

	par = params.Parameters(args)
	create_pose_data()
	
	# Calculate RGB means of images in training videos
	train_video = ['00']
	image_path_list = []
	for folder in train_video:
		image_path_list += glob.glob('Unreal/test_image_7/image_left/{}/*.png'.format(folder))
	# calculate_rgb_mean_std(image_path_list, minus_point_5=True)
	calculate_rgb_mean_std(image_path_list, minus_point_5=True)