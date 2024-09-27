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
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration



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

	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help='LLaVA pretrained model') 
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cpu", help='Device where to run inference. Default is cuda, if not found use cpu.') 

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
		
	
	model_id= args.model
	
	device= args.device
	if "cuda" in device:
		if not torch.cuda.is_available():
			print("WARN: cuda not available, using cpu...")
			device= "cpu"
	
	print('device:',device)
	
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load the model in half-precision
	if "cuda" in device:
		model = LlavaOnevisionForConditionalGeneration.from_pretrained(
			model_id, 
			torch_dtype=torch.float16, 
			device_map="auto"
		).to(device)
	else:
		model = LlavaOnevisionForConditionalGeneration.from_pretrained(
			model_id, 
			torch_dtype=torch.float16
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
	inputs = processor(images=[image_stop, image_cats], text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16).to(device)

	# Generate
	print("INFO: Generate model response ...")
	generate_ids = model.generate(**inputs, max_new_tokens=30)
	
	print("INFO: batch_decode ...")
	output= processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	#['user\n\nWhat is shown in this image?\nassistant\nThere is a red stop sign in the image.\nuser\n\nWhat about this image? How many cats do you see?\nassistant\ntwo', 'user\n\nWhat is shown in this image?\nassistant\n']
	
	print("output")
	print(output)

	
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
