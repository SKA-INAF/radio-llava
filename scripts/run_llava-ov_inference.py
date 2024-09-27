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

	# - Data options
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Apply sigma clipping transform (default=false)')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	parser.set_defaults(set_zero_to_min=False)
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	
	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help='LLaVA pretrained model') 
	
	# - Data conversation options
	parser.add_argument('--shuffle_label_options', dest='shuffle_label_options', action='store_true',help='Shuffle label options (default=false)')	
	parser.set_defaults(shuffle_labels=False)
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
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
		print("ERROR: Failed to get and parse options (err=%s)",str(ex))
		return 1
		
	# - Input filelist
	if args.inputfile=="":
		print("ERROR: Empty datalist input file!")
		return 1	
	
	inputfile= args.inputfile
	inputfile_context= args.inputfile_context
	model_id= args.model
	outfile= args.outfile
	
	device= args.device
	if "cuda" in device:
		if not torch.cuda.is_available():
			print("WARN: cuda not available, using cpu...")
			device= "cpu"
	
	print('device:',device)
	
	#===========================
	#==   READ DATALIST
	#===========================
	# - Read inference data filelist
	#print("INFO: Read image dataset filelist %s ..." % (inputfile))
	#fp= open(inputfile, "r")
	#datalist= json.load(fp)["data"]
	#nfiles= len(datalist)
	#print("INFO: #%d images present in file %s " % (nfiles, inputfile))
	
	# - Read context data filelist?
	#datalist_context= None 
	#if inputfile_context!="":
	#	print("INFO: Read image context dataset filelist %s ..." % (inputfile_context))
	#	fp= open(inputfile_context, "r")
	#	datalist_context= json.load(fp)["data"]
	#	nfiles_context= len(datalist_context)
	#	print("INFO: #%d images present in context file %s " % (nfiles_context, inputfile_context))
	
	# - Convert data filelist in conversation data
	#convert_data_to_conversations()
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load the model in half-precision
	model_id= "llava-hf/llava-onevision-qwen2-7b-ov-hf"
	model = LlavaOnevisionForConditionalGeneration.from_pretrained(
		model_id, 
		torch_dtype=torch.float16, 
		device_map="auto"
	)

	# - Load processor
	print("INFO: Load processor ...")
	processor = AutoProcessor.from_pretrained(model_id)

	# Get three different images
	print("INFO: Download images ...")
	url = "https://www.ilankelman.org/stopsigns/australia.jpg"
	image_stop = Image.open(requests.get(url, stream=True).raw)

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image_cats = Image.open(requests.get(url, stream=True).raw)

	url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
	image_snowman = Image.open(requests.get(url, stream=True).raw)

	# Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
	conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There is a red stop sign in the image."},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What about this image? How many cats do you see?"},
            ],
    },
	]

	conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
            ],
    },
	]

	prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
	prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
	
	prompts= [prompt_1]
	#prompts = [prompt_1, prompt_2]

	# We can simply feed images in the order they have to be used in the text prompt
	print("INFO: Process images ...")
	#inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
	inputs = processor(images=[image_stop, image_cats], text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)

	# Generate
	print("INFO: Generate model response ...")
	generate_ids = model.generate(**inputs, max_new_tokens=30)
	output= processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	#['user\n\nWhat is shown in this image?\nassistant\nThere is a red stop sign in the image.\nuser\n\nWhat about this image? How many cats do you see?\nassistant\ntwo', 'user\n\nWhat is shown in this image?\nassistant\n']
	
	print("output")
	print(output)

	#===========================
	#==   RUN MODEL INFERENCE
	#===========================

	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
