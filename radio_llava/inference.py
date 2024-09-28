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

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
import skimage
from PIL import Image

## TORCH MODULES
import torch

## DRAW MODULES
import matplotlib.pyplot as plt

## RADIO-LLAVA MODULES
from radio_llava.utils import *
from radio_llava.metrics import *

## LOGGER
logger = logging.getLogger(__name__)


#############################
##   INFERENCE UTILS
#############################
def run_rgz_data_inference(datalist, model, processor, device, resize_size, apply_zscale, shuffle_label_options=False, verbose=False):
	""" Convert RGZ datalist to conversational data """

	# - Define message
	context= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component (1C) radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	
	question_prefix= "Which of these morphological classes of radio sources do you see in the image? "
	question_subfix= "Please report only one identified class label. Report just NONE if you cannot recognize any of the above classes in the image."
	
	labels= ["1C-1P","1C-2P","1C-3P","2C-2P","2C-3P","3C-3P"]
	nclasses= len(labels)
	
	# - Loop over images in dataset
	for item in datalist:
		# - Get image info
		filename= item["filepaths"][0]
		class_id= item["id"]
		label= item["label"]
		
		# - Read image into PIL
		image= load_img_as_pil_rgb(
			filename,
			resize=True, resize_size=resize_size, 
			apply_zscale=apply_zscale, contrast=0.25,
			verbose=False
		)
		
		# - Create question
		option_choices= labels.copy()
		if shuffle_label_options:
			random.shuffle(option_choices)
		
		question_labels= ' \n '.join(option_choices)
		question= context + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
		
		# - Create conversation
		conversation = [
			{
				"role": "user",
				"content": [
					{"type": "image"},
					{"type": "text", "text": question},
				],
    	},
		]
		
		# - Create prompt & model inputs
		prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
		inputs = processor(image, prompt, return_tensors="pt").to(model.device, torch.float16)
		
		if verbose:
			print("inputs")
			print(inputs)
			print("inputs.pixel_values")
			print(inputs['pixel_values'].shape)

		# - Autoregressively complete prompt
		output = model.generate(
			**inputs, 
			max_new_tokens=100,
			do_sample=False
		)
		
		# - Decode response
		output_parsed= processor.decode(output[0], skip_special_tokens=True)
		output_parsed_list= output_parsed.split("assistant")

		if verbose:
			print("output")
			print(output)

			print("output_parsed")
			print(output_parsed)
		
			print("output_parsed (split assistant)")
			print(output_parsed_list)
		
		# - Extract predicted label
		label_pred= output_parsed_list[1].strip("\n").strip()

		logger.info("--> label=%s, label_pred=%s" % (label, label_pred))

		# - Compute metrics 
		# ...
		# ...

		
	return 0
	





