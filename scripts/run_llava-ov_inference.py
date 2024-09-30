#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
import io

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv
import json
import pickle

## PYTORCH
import requests
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

import matplotlib.pyplot as plt

## MODULE
from radio_llava.utils import *
from radio_llava.inference_llava import *

## LOGGER
logger = logging.getLogger(__name__)

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Path to file with image datalist (.json)') 
	parser.add_argument('-inputfile_context','--inputfile_context', dest='inputfile_context', required=False, default="", type=str, help='Path to file with image datalist for context (.json)') 

	# - Benchmark type
	parser.add_argument('-benchmark','--benchmark', dest='benchmark', required=False, default="smorph-rgz", type=str, help='Type of benchmark to run') 

	# - Data options
	parser.add_argument('--shuffle', dest='shuffle', action='store_true',help='Shuffle image data list (default=false)')	
	parser.set_defaults(shuffle=False)
	
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of entries processed in filelist (-1=all)') 
	parser.add_argument('--resize', dest='resize', action='store_true',help='Resize input image (default=false)')	
	parser.set_defaults(resize=False)
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Apply sigma clipping transform (default=false)')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--contrast', default=0.25, type=float, help='zscale contrast (default=0.25)')
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	parser.set_defaults(set_zero_to_min=False)
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	
	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help='LLaVA pretrained model') 
	
	# - Inference options
	parser.add_argument('--do_sample', dest='do_sample', action='store_true',help='Sample model response using temperature option (default=false)')	
	parser.set_defaults(do_sample=False)
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, default=0.2, type=float, help='Temperature parameter') 
	parser.add_argument('-top_p','--top_p', dest='top_p', required=False, default=None, type=float, help='top_p parameter') 
	
	# - Data conversation options
	parser.add_argument('--shuffle_label_options', dest='shuffle_label_options', action='store_true',help='Shuffle label options (default=false)')	
	parser.set_defaults(shuffle_labels=False)
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Enable verbose printout (default=false)')	
	parser.set_defaults(verbose=False)
	
	# - Outfile option
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	args = parser.parse_args()	

	return args


##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	print("INFO: Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)", str(ex))
		return 1
		
	# - Input filelist
	if args.inputfile=="":
		logger.error("Empty datalist input file!")
		return 1	
	
	inputfile= args.inputfile
	inputfile_context= args.inputfile_context
	model_id= args.model
	outfile= args.outfile
	
	device= args.device
	if "cuda" in device:
		if not torch.cuda.is_available():
			logger.warn("cuda not available, using cpu...")
			device= "cpu"
	
	logger.info("device: %s" % (device))
	
	#===========================
	#==   READ DATALIST
	#===========================
	# - Read inference data filelist
	logger.info("Read image dataset filelist %s ..." % (inputfile))
	datalist= read_datalist(inputfile)
	if args.shuffle:
		random.shuffle(datalist)
		
	nfiles= len(datalist)
	logger.info("#%d images present in file %s " % (nfiles, inputfile))
	
	# - Read context data filelist?
	datalist_context= None 
	if inputfile_context!="":
		logger.info("Read image context dataset filelist %s ..." % (inputfile_context))
		datalist_context= read_datalist(inputfile_context)
		nfiles_context= len(datalist_context)
		logger.info("#%d images present in context file %s " % (nfiles_context, inputfile_context))
		
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load the model in half-precision
	logger.info("Loading model %s ..." % (model_id))
	model, processor= load_llavaov_model(model_id, device)
	
	#===========================
	#==   RUN MODEL INFERENCE
	#===========================
	if args.benchmark=="smorph-rgz":
		logger.info("Running smorph-rgz benchmark inference ...")
		run_llavaov_model_rgz_inference(
			datalist=datalist,
			model=model,
			processor=processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_label_options=args.shuffle_label_options,
			nmax=args.nmax,
			verbose=args.verbose
		)
		
	elif args.benchmark=="smorph-radioimg":
		logger.info("Running smorph-radioimg benchmark inference")
		run_llavaov_model_smorph_inference(
			datalist=datalist,
			model=model,
			processor=processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_label_options=args.shuffle_label_options,
			nmax=args.nmax,
			verbose=args.verbose
		)	
		
	else:
		logger.error("Unknown/invalid benchmark (%s) given!" % (args.benchmark))
		return 1

	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
