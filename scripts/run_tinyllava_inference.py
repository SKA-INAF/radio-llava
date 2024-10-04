#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import numpy as np
import json
import logging
import logging.config
import argparse
import random
import warnings

# - PIL
from PIL import Image
from io import BytesIO

# - SKIMAGE/SKLEARN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss

# - TORCH
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import pipeline
from transformers import TextIteratorStreamer
	
#from tinyllava.eval.run_tiny_llava import eval_model
#from tinyllava.model.load_model import load_pretrained_model
from tinyllava.model.load_model import load_base_ckp_for_lora
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig

from tinyllava.utils.eval_utils import disable_torch_init
from tinyllava.utils.constants import *
from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.data.image_preprocess import ImagePreprocess
from tinyllava.utils.message import Message
from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
from peft import PeftModel

import matplotlib.pyplot as plt

## MODULE
from radio_llava.utils import *
from radio_llava.inference_tinyllava import *

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
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B", help='LLaVA pretrained model') 
	parser.add_argument('--load_lora_model', dest='load_lora_model', action='store_true',help='Load model LORA (default=false)')	
	parser.set_defaults(load_lora_model=False)
	parser.add_argument('--reset_imgnorm', dest='reset_imgnorm', action='store_true',help='Reset vision model image normalization to mean=0, std=1 (default=false)')	
	parser.set_defaults(reset_imgnorm=False)
	
	# - Inference options
	parser.add_argument('--do_sample', dest='do_sample', action='store_true',help='Sample model response using temperature option (default=false)')	
	parser.set_defaults(do_sample=False)
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, default=0.2, type=float, help='Temperature parameter') 
	parser.add_argument('-top_p','--top_p', dest='top_p', required=False, default=None, type=float, help='top_p parameter') 
	parser.add_argument('-conv_mode','--conv_mode', dest='conv_mode', required=False, default="phi", type=str, help='conv_mode inference par') 
	
	# - Data conversation options
	parser.add_argument('--shuffle_options', dest='shuffle_options', action='store_true',help='Shuffle task options (default=false)')	
	parser.set_defaults(shuffle_options=False)
	parser.add_argument('--add_task_description', dest='add_task_description', action='store_true',help='Add task description (default=false)')	
	parser.set_defaults(add_task_description=False)
	
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
	
	#===========================
	#==   LOAD MODEL
	#===========================
	logger.info("Loading model %s ..." % (model_id))
	model= load_tinyllava_model(model_id, args.load_lora_model, device)
	if model is None:
		logger.error("Failed to load model %s ..." % (model_id))
		return 1
	
	#===========================
	#==   RUN MODEL INFERENCE
	#===========================
	logger.info("Running %s benchmark inference ..." % (args.benchmark))

	if args.benchmark=="smorph-rgz":
		run_tinyllava_model_rgz_inference(
			datalist=datalist,
			model=model,
			device=device,
			reset_imgnorm=args.reset_imgnorm,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			conv_mode=args.conv_mode,
			shuffle_options=args.shuffle_options, nmax=args.nmax,
			add_task_description=args.add_task_description,
			verbose=args.verbose
		)
		
	elif args.benchmark=="smorph-radioimg":
		run_tinyllava_model_smorph_inference(
			datalist=datalist,
			model=model,
			device=device,
			reset_imgnorm=args.reset_imgnorm,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			conv_mode=args.conv_mode,
			shuffle_options=args.shuffle_options, nmax=args.nmax,
			add_task_description=args.add_task_description,
			verbose=args.verbose
		)	
	
	elif args.benchmark=="galaxydet-radioimg":
		run_tinyllava_model_galaxy_inference(
			datalist=datalist,
			model=model,
			device=device,
			reset_imgnorm=args.reset_imgnorm,
			resize=args.resize, resize_size=args.imgsize, 
			zscale=args.zscale, contrast=args.contrast,
			conv_mode=args.conv_mode,
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
		
