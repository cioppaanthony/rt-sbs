"""
Copyright (c) 2020 - University of Liège
Anthony Cioppa (anthony.cioppa@uliege.be), University of Liège (ULiège), Montefiore Institute, TELIM.
All rights reserved - patented technology, software available under evaluation license (see LICENSE)
"""

import torch
import utils.median_pool as mp
import torchvision
import cv2

class ConfusionMatrix:

	def __init__(self, device, median_kernel_size = None):

		self.TP = 0
		self.FP = 0
		self.FN = 0
		self.TN = 0

		self.GROUNDTRUTH_BG = 65
		self.GROUNDTRUTH_FG = 220 
		self.BGS_THRESHOLD = 127

		self.ones = None 
		self.zeros = None

		self.device = device

		self.MedianPool = None
		if median_kernel_size is not None:
			self.MedianPool = mp.MedianPool2d(kernel_size=int(median_kernel_size), same=True)

	def update_MedialPool(self, median_kernel_size):
		self.MedianPool = mp.MedianPool2d(kernel_size=int(median_kernel_size), same=True)

	def evaluate(self, mask, groundtruth):

		groundtruth = groundtruth.to(self.device)
		if self.MedianPool is not None:
			mask = self.MedianPool.forward(mask.unsqueeze_(0).unsqueeze_(0)).squeeze_(0).squeeze_(0)
		mask = mask*255

		if self.ones is None or self.zeros is None:
			self.update(groundtruth.size()[0], groundtruth.size()[1])

		TP_mask = torch.where((mask >= self.BGS_THRESHOLD) & (groundtruth > self.GROUNDTRUTH_FG), self.ones, self.zeros)
		FP_mask = torch.where((mask >= self.BGS_THRESHOLD) & (groundtruth < self.GROUNDTRUTH_BG), self.ones, self.zeros)
		FN_mask = torch.where((mask < self.BGS_THRESHOLD) & (groundtruth > self.GROUNDTRUTH_FG), self.ones, self.zeros)
		TN_mask = torch.where((mask < self.BGS_THRESHOLD) & (groundtruth < self.GROUNDTRUTH_BG), self.ones, self.zeros)
		
		self.TP += torch.sum(TP_mask)
		self.FP += torch.sum(FP_mask)
		self.FN += torch.sum(FN_mask)
		self.TN += torch.sum(TN_mask)

	def save_image(self, path, mask, frame_index):

		if self.MedianPool is not None:
			mask = self.MedianPool.forward(mask.unsqueeze_(0).unsqueeze_(0)).squeeze_(0).squeeze_(0)
		mask = mask*255

		TP_mask = torch.where((mask > 0) & (mask <= 255), self.ones, self.zeros)
		print(torch.sum(TP_mask))

		frame_numpy = (mask.to("cpu").numpy()).astype("uint8")
		frame_path = path + "/bin" + (str(frame_index+1)).zfill(6) + ".png"
		cv2.imwrite(frame_path, frame_numpy)

	def update(self, height, width):

		self.ones = torch.ones((height, width), dtype=torch.float32, device=self.device)
		self.zeros = torch.zeros((height, width), dtype=torch.float32, device=self.device)
	
	def F1(self):

		return (2*self.TP)/(2*self.TP + self.FP + self.FN)

	def to_device(self, array):

		return torch.from_numpy(array).to(self.device).type(torch.float32)

class ConfusionMatricesHolder:

	def __init__(self, device, categories, videos, median_kernel_size=None):

		self.device = device

		self.confusion_matrix = dict()

		for category in categories:

			self.confusion_matrix[category] = dict()

			for video in videos[category]:

				self.confusion_matrix[category][video] = ConfusionMatrix(device, median_kernel_size)

	def meanF1(self, categories, videos):

		meanF1 = 0

		for category in categories:

			meanF1Cat = 0

			for video in videos[category]:

				meanF1Cat += self.confusion_matrix[category][video].F1()

			meanF1 += meanF1Cat/len(videos[category])

		return meanF1/len(categories)