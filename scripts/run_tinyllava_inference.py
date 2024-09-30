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
from radio_llava.inference import *

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
	#parser.add_argument('-inputfile_context','--inputfile_context', dest='inputfile_context', required=False, default="", type=str, help='Path to file with image datalist for context (.json)') 

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
## MODEL
##############
def load_model_lora(model_name_or_path):
	""" Create LORA model from config and load weights """

	# - Check if adapter file exists
	adapter_file= os.path.join(model_name_or_path, 'adapter_config.json') 
	if not os.path.exists(adapter_file):
		logger.error("Cannot find adapted config file %s in model path!" % (adapter_file))
		return None

	# - Build model
	model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
	model = TinyLlavaForConditionalGeneration(model_config)
	
	# - Build LLM model and load weights
	language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
	language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)		
	model.language_model.load_state_dict(language_model_ckp)
	
	# - Build vision model and load weights
	vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
	vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
	model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
		
	# - Build connector model and load weights
	connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
	connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
	model.connector.load_state_dict(connector_ckp)
	
	# - Set model float16	
	model.to(torch.float16)
	
	# - Merge LORA weights into model
	logger.debug("Loading LoRA weights...")
	model = PeftModel.from_pretrained(model, model_name_or_path)
	logger.debug("Merging LoRA weights...")
	model = model.merge_and_unload()
	logger.debug("Model is loaded...")
	
	return model
		

def load_model(model_name_or_path):
	""" Create model from config and load weights """
	
	# - Load config & model
	logger.debug("Read tinyllava config ...")
	model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
            
	logger.debug("Initialize tinyllava model from config ...")
	model = TinyLlavaForConditionalGeneration(model_config)
            
	logger.debug("Set model component weights paths ...")
	language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
	vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
	connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
            
	language_model_path = os.path.join(model_name_or_path, 'language_model')
	vision_tower_path = os.path.join(model_name_or_path, 'vision_tower')
	connector_path = os.path.join(model_name_or_path, 'connector')
            
	# - Load connector weights
	logger.debug("Loading connector weights from file %s ..." % (connector_ckp_path))
	model.load_connector(model_name_or_path=connector_path)
            
	# - Load LLM weights
	logger.debug("Loading LLM weights ...")
	model.load_llm(model_name_or_path=language_model_path)
            
	# - Load vision weights
	model.load_vision_tower(model_name_or_path=vision_tower_path)
       
	# - Set model float16	     
	model.to(torch.float16)

	return model


def load_pretrained_model(
	model_name_or_path,
	load_lora_model=False
):
	""" Load TinyLLaVA model """
    
	# - Check model name
	if model_name_or_path is None or model_name_or_path=="":
		logger.error("Empty or invalid model path given!")
		return None
		
	has_lora_in_name= ('lora' in model_name_or_path)
	if has_lora_in_name and not load_lora_model:    
		logger.warn("lora was found in model name but load_lora_model is False, will load as standard model but check if this is really the desired behavior!")
    
 	# - Load model from path
 	if load_lora_model:
 		model= load_model_lora(model_name_or_path)
 	else:
 		try:
			model = TinyLlavaForConditionalGeneration.from_pretrained(
				model_name_or_path,
				low_cpu_mem_usage=True,
				torch_dtype=torch.float16,
				device_map="auto"
		)
 		except Exception as e:
			logger.warn("Failed to load pre-trained model (err=%s), trying with another method ..." % (str(e)))
			model= load_model(model_name_or_path)
 	
 	# - Check model
 	if model is None:
 		logger.error("Failed to load model %s!" % (model_name_or_path))
 		return None
 	
 	# - Set model options
 	image_processor = model.vision_tower._image_processor
	context_len = getattr(model.config, 'max_sequence_length', 2048)
	tokenizer = model.tokenizer
	
	return model, tokenizer, image_processor, context_len

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
	
	#===========================
	#==   LOAD MODEL
	#===========================
	logger.info("Loading model %s ..." % (model_id))
	res= load_pretrained_model(model_id, args.load_lora_model)
	if res is None:
		logger.error("Failed to load model %s ..." % ())
		return 1
	model, tokenizer, image_processor, context_len= res
		
	if args.reset_imgnorm:
		model.vision_tower._image_processor.image_mean= [0.,0.,0.]
		model.vision_tower._image_processor.image_std= [1.,1.,1.]
	
	#===========================
	#==   RUN MODEL INFERENCE
	#===========================
	# ...	
		
		
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())	
		
