"""
Copyright (c) 2020 - University of Liège
Anthony Cioppa (anthony.cioppa@uliege.be), University of Liège (ULiège), Montefiore Institute, TELIM.
All rights reserved - patented technology, software available under evaluation license (see LICENSE)
"""

import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
import json
import torch

def getCategories(dataset_dir):
	"""
	Stores the list of categories as string and the videos of each
	category in a dictionary.
	"""

	categories = sorted(os.listdir(dataset_dir), key=lambda v: v.upper())

	videos = dict()

	for category in categories:
		category_dir = os.path.join(dataset_dir, category)
		videos[category] = sorted(os.listdir(category_dir), key=lambda v: v.upper())

	return categories, videos

def readVideo(video_dir, img_type, num_channels, data_type, save_dir = None):
	"""
	Read a video as a set of frames and stores them in a numpy array
	"""
	frames_path = []
	frames_path.extend(sorted(glob.glob(os.path.join(video_dir, "*.png"))))
	frames_path.extend(sorted(glob.glob(os.path.join(video_dir, "*.jpg"))))
	
	num_frames = len(frames_path)

	frame_calibration = cv2.imread(frames_path[0], img_type)
	height = frame_calibration.shape[0]
	width = frame_calibration.shape[1]

	array = np.zeros((num_frames,) + frame_calibration.shape, dtype=data_type)

	counter = 0
	for frame_path in tqdm(frames_path):
		array[counter] = cv2.imread(frame_path, img_type).astype(data_type)
		counter += 1

	return array