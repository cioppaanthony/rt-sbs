"""
Copyright (c) 2020 - University of Liège
Anthony Cioppa (anthony.cioppa@uliege.be), University of Liège (ULiège), Montefiore Institute, TELIM.
All rights reserved - patented technology, software available under evaluation license (see LICENSE)
"""

import sys
import os
import cv2
import time
import torch
from tqdm import tqdm
import utils.evaluate as evaluate
import utils.read as read
import algorithms.ViBeGPU as ViBeGPU
import algorithms.rtsbs as rtsbs

from utils.argument_parser import args

# Definition of the device
device = torch.device(args.device)

# Get the names of the categories and the videos
categories, videos = read.getCategories(args.dataset)

# Create the evaluation confusion matrices
confusion_matrices = evaluate.ConfusionMatricesHolder(device, categories, videos, args.median)

# Loop over all categories that were retrieved
for category in categories:

	# Loop over all videos
	for video in videos[category]:

		# Definition of the video directory path
		video_input_dir = os.path.join(args.dataset, category, video, "input")

		# Definition of the groundtruth video directory path
		video_groundtruth_dir = os.path.join(args.dataset, category, video, "groundtruth")

		# Definition of the semantic video directory path
		video_semantic_dir = os.path.join(args.semantic, category, video)

		# Defition of the arrays that will contain the different sequences
		print("Loading the input video")
		video_original = read.readVideo(video_input_dir, img_type=cv2.IMREAD_COLOR, num_channels=3 , data_type="uint8")
		video_original = torch.from_numpy(video_original).transpose(1,3).transpose(2,3).type(torch.float)
		print("Loading the ground-truth")
		video_groundtruth = read.readVideo(video_groundtruth_dir, img_type=cv2.IMREAD_GRAYSCALE, num_channels=1 , data_type="uint8")
		video_groundtruth = torch.from_numpy(video_groundtruth).type(torch.float)
		print("Loading the semantic masks")
		video_semantic = read.readVideo(video_semantic_dir, img_type=cv2.IMREAD_ANYDEPTH, num_channels=1 , data_type="float32")
		video_semantic = torch.from_numpy(video_semantic).type(torch.float)

		print("Processing of the video")

		# At this point, the videos are loaded in their correct format
		# The individual frames will be transfered to the GPU and transformed to float for operations

		# Creation of the rt-sbs class
		frame_init = video_original[0].to(device)
		bgs = ViBeGPU.ViBe(frame_init, device)
		frame_init = frame_init.to("cpu")

		# Creation of the rt-sbs class
		semantic_processing = rtsbs.RTSBS(device, video_semantic[0], args.taubg, args.taufg, args.taubgstar, args.taufgstar, args.moduloupdate)

		# Frame index to know at which frame of the video we are
		frame_index = 0
		frame_rtsbs = None

		time_start = time.time()

		#Loop over all frames of the video
		p_bar = tqdm(total = video_original.shape[0])
		for frame_original, frame_groundtruth, frame_semantic in zip(video_original, video_groundtruth, video_semantic):
			
			

			frame_original = frame_original.to(device)
			frame_semantic = frame_semantic.to(device)
			
			mask = bgs.segmentation_(frame_original)
			
			# With semantics
			if frame_index %args.framerate == 0:
				frame_rtsbs = semantic_processing.segment_semantics(frame_original, mask, frame_semantic)

			# Without semantics
			else:
				frame_rtsbs = semantic_processing.segment_no_semantics(frame_original, mask)

			bgs.update_(frame_original, frame_rtsbs)

			confusion_matrices.confusion_matrix[category][video].evaluate(frame_rtsbs, frame_groundtruth)

			frame_original = frame_original.to("cpu")
			mask = mask.to("cpu")
			
			p_bar.update(1)
			
			frame_index += 1
		p_bar.close()

		time_stop = time.time()
		print(category + " - ", video)
		print("F1: ", confusion_matrices.confusion_matrix[category][video].F1())
		print("Timing with device: ", device, " = ", time_stop-time_start, " seconds")

print("Mean F1 Score", confusion_matrices.meanF1(categories, videos))
"""
file = open("../output/log-rtsbs.log",'a')
file.write(str(args.framerate))
file.write(": ")
file.write(str(confusion_matrices.meanF1(categories, videos)))
file.write("\n")
file.close()
"""