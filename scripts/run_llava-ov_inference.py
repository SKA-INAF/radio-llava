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
import re

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=False, type=str, help='Path to file with image datalist (.json)') 
	parser.add_argument('-inputfile_context','--inputfile_context', dest='inputfile_context', required=False, default="", type=str, help='Path to file with image datalist for context (.json)') 
	parser.add_argument('-image','--image', dest='image', required=False, default="", type=str, help='Path to image file') 

	# - Benchmark type
	parser.add_argument('-benchmark','--benchmark', dest='benchmark', required=False, default="", type=str, help='Type of benchmark to run {smorph-rgz,smorph-radioimg,galaxydet-radioimg,artefactdet-radioimg,anomalydet-radioimg,galaxymorphclass-mirabest,galaxymorphclass-gmnist}') 
	parser.add_argument('-prompts','--prompts', dest='prompts', required=False, default="", type=str, help='Prompts text to be given to model, separated by <END> separator') 

	# - Data options
	parser.add_argument('--shuffle', dest='shuffle', action='store_true',help='Shuffle image data list (default=false)')	
	parser.set_defaults(shuffle=False)
	parser.add_argument('--shuffle_context', dest='shuffle_context', action='store_true',help='Shuffle context data (default=false)')	
	parser.set_defaults(shuffle_context=False)
	
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of entries processed in filelist (-1=all)') 
	parser.add_argument('-nmax_context','--nmax_context', dest='nmax_context', required=False, type=int, default=-1, help='Max number of entries processed in context filelist (-1=all)') 
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
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-ov", help='LLaVA pretrained model') 
	parser.add_argument('-conv_template','--conv_template', dest='conv_template', required=False, type=str, default="qwen_1_5", help='LLaVA conversation template') 
	parser.add_argument('-model_name','--model_name', dest='model_name', required=False, type=str, default="llava_qwen", help='LLaVA model name') 
	parser.add_argument('-model_base','--model_base', dest='model_base', required=False, type=str, default=None, help='Base model name (needed for LORA & for adapter-only fine-tuning)') 
	
	# - Inference options
	parser.add_argument('--do_sample', dest='do_sample', action='store_true',help='Sample model response using temperature option (default=false)')	
	parser.set_defaults(do_sample=False)
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, default=0.2, type=float, help='Temperature parameter') 
	parser.add_argument('-top_p','--top_p', dest='top_p', required=False, default=None, type=float, help='top_p parameter') 
	parser.add_argument('-max_new_tokens','--max_new_tokens', dest='max_new_tokens', required=False, type=int, default=4096, help='Max new token parameter') 
	
	
	# - Data conversation options
	parser.add_argument('--shuffle_options', dest='shuffle_options', action='store_true', help='Shuffle label options (default=false)')	
	parser.set_defaults(shuffle_options=False)
	parser.add_argument('--add_task_description', dest='add_task_description', action='store_true',help='Add task description (default=false)')	
	parser.set_defaults(add_task_description=False)
	parser.add_argument('-prompt_version','--prompt_version', dest='prompt_version', required=False, type=str, default="v1", help='Prompt version {v1,v2}') 
	
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
	image_path= args.image
	
	prompts= re.split("<END>", args.prompts)
	prompts= [item.strip("\"'") for item in prompts]
	prompts_escape= [item.encode().decode('unicode_escape').strip("\"'") for item in prompts]
	
	#prompt_escape = args.prompt.encode().decode('unicode_escape').strip("\"'")
	#prompt = args.prompt.strip("\"'")
	
	if inputfile=="" and image_path=="":
		logger.error("inputfile and image are empty, you must specify at least one!")
		return 1
		
	if image_path!="" and prompts=="":
		logger.error("Prompts is empty, you must specify a prompt text for single-image input!")
		return 1
	
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
	# - Read inference data filelist if image option not given
	if image_path=="":	
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
			if args.shuffle_context:
				random.shuffle(datalist_context)
			nfiles_context= len(datalist_context)
			logger.info("#%d images present in context file %s " % (nfiles_context, inputfile_context))
		
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load the model in half-precision
	logger.info("Loading model %s ..." % (model_id))
	model, tokenizer, image_processor= load_llavaov_model(
		model_id, 
		model_name=args.model_name,
		model_base=args.model_base, 
		device_map="auto"
	)
	
	#===========================
	#==   RUN MODEL INFERENCE
	#===========================
	if args.benchmark=="smorph-rgz":
		logger.info("Running smorph-rgz benchmark inference ...")
		run_llavaov_model_rgz_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
		
	elif args.benchmark=="smorph-radioimg":
		logger.info("Running smorph-radioimg benchmark inference ...")
		run_llavaov_model_smorph_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
	
	elif args.benchmark=="galaxydet-radioimg":
		logger.info("Running galaxydet-radioimg benchmark inference ...")
		run_llavaov_model_galaxy_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
		
	elif args.benchmark=="artefactdet-radioimg":
		logger.info("Running artefactdet-radioimg benchmark inference ...")
		run_llavaov_model_artefact_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
		
	elif args.benchmark=="anomalydet-radioimg":
		logger.info("Running anomalydet-radioimg benchmark inference ...")
		run_llavaov_model_anomaly_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
		
	elif args.benchmark=="galaxymorphclass-mirabest":
		logger.info("Running galaxymorphclass-mirabest benchmark inference ...")
		run_llavaov_model_mirabest_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
	
	elif args.benchmark=="galaxymorphclass-gmnist":
		logger.info("Running galaxymorphclass-gmnist benchmark inference ...")
		run_llavaov_model_gmnist_inference(
			datalist=datalist,
			model=model,
			tokenizer=tokenizer,
			image_processor=image_processor,
			datalist_context=datalist_context,
			device=device,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			shuffle_options=args.shuffle_options,
			nmax=args.nmax,
			nmax_context=args.nmax_context,
			add_task_description=args.add_task_description,
			conv_template=args.conv_template,
			prompt_version=args.prompt_version,
			verbose=args.verbose
		)
		
	else:
		#logger.error("Unknown/invalid benchmark (%s) given!" % (args.benchmark))
		#return 1

		for i, prompt in enumerate(prompts):
			logger.info("Running inference on image %s ..." % (image_path))
			response= run_llavaov_model_inference_on_image(
				image_path,
				model=model, 
				tokenizer=tokenizer, 
				image_processor=image_processor,
				prompt=prompt,
				resize=args.resize, resize_size=args.imgsize, 
				zscale=args.zscale, contrast=args.contrast,
				do_sample=args.do_sample,
				temperature=args.temperature,
				max_new_tokens=args.max_new_tokens,
				conv_template=args.conv_template,
				verbose=args.verbose
			)
			if response is not None:
				print("== QUESTION (raw) ==")
				print(prompt)
				print("== QUESTION ==")
				print(prompts_escape[i])
				print("== ANSWER ==")
				print(response)
			else:
				logger.warning("Inference failed")
				continue

	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
