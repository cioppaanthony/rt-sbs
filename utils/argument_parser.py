"""
Copyright (c) 2020 - University of Liège
Anthony Cioppa (anthony.cioppa@uliege.be), University of Liège (ULiège), Montefiore Institute, TELIM.
All rights reserved - patented technology, software available under evaluation license (see LICENSE)
"""

import argparse

parser = argparse.ArgumentParser()

# Input arguments
parser.add_argument("-d", "--dataset", help="filepath to the CDNet dataset")
parser.add_argument("-ds", "--semantic", help="filepath to the PSPNet segmentation folder")
parser.add_argument("-de", "--device", help="device on which to run", type=str, default="cuda:0")
parser.add_argument("-f", "--framerate", help="semantic framerate (one out of X frames)", type=int, default=5)

# SBS and RT-SBS thresholds
parser.add_argument("-tbgs", "--taubgstar", help="tau_bg* in the paper (8bits)", type=int, default=65)
parser.add_argument("-tfgs", "--taufgstar", help="tau_fg* in the paper (8bits)", type=int, default=115)
parser.add_argument("-tbg", "--taubg", help="tau_bg in the paper (16bits)", type=int, default=300)
parser.add_argument("-tfg", "--taufg", help="tau_fg in the paper (8bits)", type=int, default=175)

# Extra SBS arguments
parser.add_argument("-mu", "--moduloupdate", help="modulo update", type=int, default=256)
parser.add_argument("-m", "--median", help="size of kernel of the median filtering", default = 9)

args = parser.parse_args()