"""
Copyright (c) 2020 - University of Liège
Anthony Cioppa (anthony.cioppa@uliege.be), University of Liège (ULiège), Montefiore Institute, TELIM.
All rights reserved - patented technology, software available under evaluation license (see LICENSE)
"""

import torch
import numpy as np
import torchvision


class ViBe:

	def __init__(self, image, device, numberOfSamples = 30, matchingThreshold = 10, matchingNumber = 2, updateFactor = 8.0, neighborhood_radius = 1):

		# Parameters related to the video size
		self.channels = image.size()[0]
		self.height = image.size()[1]
		self.width = image.size()[2]

		self.device = device

		# Parameters related to the method itself
		self.numberOfSamples = numberOfSamples
		self.matchingThreshold = matchingThreshold
		self.matchingNumber = matchingNumber
		self.updateFactor = updateFactor

		# Storage for the history
		self.historyBuffer = torch.zeros(image.size()[0], image.size()[1], image.size()[2], self.numberOfSamples, dtype = torch.float, device = self.device)

		# Buffers with random values
		self.update = torch.empty(self.width*self.height, dtype = torch.float).uniform_(0,1).to(self.device)
		self.neighbor_row = None
		self.neighbor_col = None
		self.position = None

		# Some other precomputations
		self.row = torch.arange(0, self.height, dtype = torch.float, device = self.device).repeat(self.width,1).transpose(0,1)
		self.col = torch.arange(0, self.width, dtype = torch.float, device = self.device).repeat(self.height,1)

		# Threshold values
		self.one = torch.zeros(self.update.size(), dtype = torch.float, device = self.device) + 1
		self.zero = torch.zeros(self.update.size(), dtype = torch.float, device = self.device)
		self.BG = torch.zeros(image.size(), dtype = torch.float, device = self.device)
		self.FG = torch.zeros(image.size(), dtype = torch.float, device = self.device) + 255
		self.BG1 = torch.zeros(image.size()[1:3], dtype = torch.float, device = self.device)
		self.FG1 = torch.zeros(image.size()[1:3], dtype = torch.float, device = self.device) + 1


		# --------------------
		# Some initializations
		# --------------------
		"""
			We want to guarantee at least two matches in each pixel just after
			the initialization, in order to predict only background. So, the
			random noise (this idea originates from the C implementation by
			Marc Van Droogenbroeck) is only added on numberOfSamples-matchingNumber
			samples.
		""" 

		for test in np.arange(self.matchingNumber):
			self.historyBuffer[:,:,:,test] = image

		for test in np.arange(self.numberOfSamples-self.matchingNumber) + self.matchingNumber:
			noise = torch.randint(-20,20,image.size()).to(self.device).type(torch.float)
			value_plus_noise = image + noise
			value_plus_noise = torch.where(value_plus_noise > 255, self.FG, value_plus_noise)
			value_plus_noise = torch.where(value_plus_noise < 0, self.BG, value_plus_noise)
			self.historyBuffer[:,:,:,test] = value_plus_noise

		self.update = torch.where(self.update > 1/self.updateFactor, self.zero, self.one)


		amount = int(torch.sum(self.update).to("cpu").numpy())

		self.neighbor_row = torch.randint(-neighborhood_radius, neighborhood_radius+1, (amount,)).to(self.device).type(torch.float)
		self.neighbor_col = torch.randint(-neighborhood_radius, neighborhood_radius+1, (amount,)).to(self.device).type(torch.float)

		self.position = torch.randint(0, self.numberOfSamples, (amount,)).to(self.device).type(torch.float)



	def segmentation_(self, image):

		num_matches = torch.zeros(image.size()[1:3], dtype = torch.float, device = self.device)
		segmentation_map = torch.zeros(image.size()[1:3], dtype = torch.float, device = self.device)

		if self.channels == 1:
			matchingThreshold = self.matchingThreshold
			for test in np.arange(self.numberOfSamples):
				delta = image - self.historyBuffer[:,:,:,test]
				match = torch.where(torch.abs(delta) <= matchingThreshold, self.FG1, self.BG1)
				num_matches = num_matches + match
			segmentation_map = torch.where(num_matches >= self.matchingNumber, self.BG1, self.FG1)
		else :
			matchingThreshold = 4.5 * self.matchingThreshold
			for test in np.arange(self.numberOfSamples):
				delta = image - self.historyBuffer[:,:,:,test]
				match = torch.where(torch.sum(torch.abs(delta),0) <= matchingThreshold, self.FG1, self.BG1)
				num_matches = num_matches + match
			segmentation_map = torch.where(num_matches >= self.matchingNumber, self.BG1, self.FG1)
		return segmentation_map

	def update_(self, image, updating_mask):


		# Perturbation of the precomputed array
		r = int(torch.randint(0,self.update.size()[0],(1,)).numpy())
		r2 = torch.randint(0,self.position.size()[0],(3,)).numpy().astype(int)

		# Rolling the vectors
		update_image = self.roll_(self.update, r).view(self.height, self.width)
		self.neighbor_row = self.roll_(self.neighbor_row, r2[0])
		self.neighbor_col = self.roll_(self.neighbor_col, r2[1])
		self.position = self.roll_(self.position, r2[2])

		update = update_image * (1-updating_mask)
		num_updates = int(torch.sum(update).to("cpu").numpy())

		if self.channels == 3:

			# Replace one value of the history, at the pixel, by the current value
			row = self.row[update == 1]
			col = self.col[update == 1]
			pos = self.position[0:num_updates]

			self.historyBuffer[:,row.type(torch.LongTensor),col.type(torch.LongTensor),pos.type(torch.LongTensor)] = image[:,row.type(torch.LongTensor),col.type(torch.LongTensor)]

			# Replace one value of the history, at a neighbor pixel, by the current value
			row_shift = row + self.neighbor_row[0:num_updates]
			row_shift = torch.where(row_shift >= self.height, self.one[0:num_updates]*self.height-1, row_shift)
			row_shift = torch.where(row_shift < 0, self.zero[0:num_updates], row_shift)

			col_shift = col + self.neighbor_col[0:num_updates]
			col_shift = torch.where(col_shift >= self.width, self.one[0:num_updates]*self.width-1, col_shift)
			col_shift = torch.where(col_shift < 0, self.zero[0:num_updates], col_shift)

			pos = self.roll_(self.position, r2[0])[0:num_updates]

			self.historyBuffer[:,row_shift.type(torch.LongTensor),col_shift.type(torch.LongTensor),pos.type(torch.LongTensor)] = image[:,row.type(torch.LongTensor),col.type(torch.LongTensor)]


	def roll_(self, x, n):  
		return torch.cat((x[-n:], x[:-n]))

	def print_(self):

		print("------------------------------------------")
		print("model.historyBuffer size : ", self.historyBuffer.size())
		print("model.historyBuffer type : ", self.historyBuffer.type())
		print("model.historyBuffer min : ", self.historyBuffer.min())
		print("model.historyBuffer max : ", self.historyBuffer.max())
		print("model.historyBuffer mean : ", self.historyBuffer.mean())
		print(self.historyBuffer)
		print()
		print("------------------------------------------")
		print("model.update size : ", self.update.size())
		print("model.update type : ", self.update.type())
		print("model.update min : ", self.update.min())
		print("model.update max : ", self.update.max())
		print("model.update mean : ", self.update.mean())
		print(self.update)
		print()
		print("------------------------------------------")
		print("model.neighbor_row size : ", self.neighbor_row.size())
		print("model.neighbor_row type : ", self.neighbor_row.type())
		print("model.neighbor_row min : ", self.neighbor_row.min())
		print("model.neighbor_row max : ", self.neighbor_row.max())
		print("model.neighbor_row mean : ", self.neighbor_row.mean())
		print(self.neighbor_row)
		print()
		print("------------------------------------------")
		print("model.neighbor_col size : ", self.neighbor_col.size())
		print("model.neighbor_col type : ", self.neighbor_col.type())
		print("model.neighbor_col min : ", self.neighbor_col.min())
		print("model.neighbor_col max : ", self.neighbor_col.max())
		print("model.neighbor_col mean : ", self.neighbor_col.mean())
		print(self.neighbor_col)
		print()
		print("------------------------------------------")
		print("model.position size : ", self.position.size())
		print("model.position type : ", self.position.type())
		print("model.position min : ", self.position.min())
		print("model.position max : ", self.position.max())
		print("model.position mean : ", self.position.mean())
		print(self.position)
		print()
		print("------------------------------------------")
		print("model.row size : ", self.row.size())
		print("model.row type : ", self.row.type())
		print("model.row min : ", self.row.min())
		print("model.row max : ", self.row.max())
		print("model.row mean : ", self.row.mean())
		print(self.row)
		print()
		print("------------------------------------------")
		print("model.col size : ", self.col.size())
		print("model.col type : ", self.col.type())
		print("model.col min : ", self.col.min())
		print("model.col max : ", self.col.max())
		print("model.col mean : ", self.col.mean())
		print(self.col)
		print()