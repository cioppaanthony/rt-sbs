"""
Copyright (c) 2020 - University of Liège
Anthony Cioppa (anthony.cioppa@uliege.be), University of Liège (ULiège), Montefiore Institute, TELIM.
All rights reserved - patented technology, software available under evaluation license (see LICENSE)
"""

import torch
import numpy as np

class RTSBS:

	def __init__(self, device, semantic, tau_BG, tau_FG, tau_BGS, tau_FGS, modulo_update):

		self.device = device

		self.tau_BG = float(tau_BG)
		self.tau_FG = float(tau_FG*256)
		self.tau_BGS = float(tau_BGS)
		self.tau_FGS = float(tau_FGS)
		self.modulo_update = float(modulo_update)

		self.height = semantic.shape[0]
		self.width = semantic.shape[1]
		self.total = 0

		self.semantic_model = semantic.to(device)
		self.rule_map_BG = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
		self.rule_map_FG = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
		self.color_map = torch.zeros((self.height, self.width, 3), dtype=torch.float32, device=self.device)

		self.ones = torch.ones((self.height, self.width), dtype=torch.float32, device=self.device)
		self.zeros = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
		self.background = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
		self.foreground = torch.ones((self.height, self.width), dtype=torch.float32, device=self.device)*1

	def segment_semantics(self, frame, mask, semantic):

		self.color_map = frame.to(self.device)

		mask_device = mask.to(self.device)

		semantic_device = semantic.to(self.device)

		# Rule map for rule 1
		self.rule_map_BG = torch.where(semantic_device <= self.tau_BG, self.ones , self.zeros)

		# Rule map for rule 2
		semantic_increase = semantic_device-self.semantic_model
		self.rule_map_FG = torch.where(semantic_increase >= self.tau_FG, self.ones, self.zeros)

		# Application of the rule maps
		mask_device = torch.where(self.rule_map_BG == 1, self.background, mask_device)
		mask_device = torch.where((self.rule_map_FG == 1) & (self.rule_map_BG == 0), self.foreground, mask_device)

		# Update of the semantic model
		update_pixels = torch.cuda.FloatTensor(self.height, self.width, device=self.device).uniform_() < (1/self.modulo_update)
		self.semantic_model = torch.where((update_pixels == 1) & (mask_device == 0), semantic_device, self.semantic_model)

		return mask_device

	def segment_no_semantics(self, frame, mask):

		mask_device = mask.to(self.device)

		frame_device = frame.to(self.device)

		# Computation of the color difference
		color_diff = torch.abs(frame_device-self.color_map)
		color_diff = torch.sum(color_diff, dim=0)

		# Application of the rule maps
		mask_device = torch.where((self.rule_map_BG == 1) & (color_diff <= self.tau_BGS), self.background, mask_device)
		mask_device = torch.where((self.rule_map_FG == 1) & (self.rule_map_BG == 0) & (color_diff <= self.tau_FGS), self.foreground, mask_device)

		return mask_device

	def to_device(self, array):

		return torch.from_numpy(array).to(self.device).type(torch.float32)

