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
from radio_llava.metrics import *

## LOGGER
logger = logging.getLogger(__name__)


######################
##   LOAD MODEL
######################
def load_tinyllava_model_lora(model_name_or_path):
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
		

def load_tinyllava_model_standard(model_name_or_path):
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


def load_tinyllava_model(
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
		model= load_tinyllava_model_lora(model_name_or_path)
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
			model= load_tinyllava_model_standard(model_name_or_path)
 	
	# - Check model
	if model is None:
		logger.error("Failed to load model %s!" % (model_name_or_path))
		return None
 	
	# - Extract image processor from model
	#logger.debug("Extracting image processor from model ...")	
	#if reset_imgnorm:
	#	model.vision_tower._image_processor.image_mean= [0.,0.,0.]
	#	model.vision_tower._image_processor.image_std= [1.,1.,1.]
	
	###image_processor = model.vision_tower._image_processor
	#image_processor = ImagePreprocess(model.vision_tower._image_processor, model.config)
	
	# - Extract tokenizer from model
	#context_len = getattr(model.config, 'max_sequence_length', 2048)
	#tokenizer = model.tokenizer
	
	# - Extract text processor from model
	#text_processor = TextPreprocess(tokenizer, conv_mode)
	
	#return model, tokenizer, image_processor, context_len
	return model
	
######################
##   INFERENCE
######################
def run_tinyllava_model_query(
	model, 
	image_path, 
	query,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	conv_mode='phi',
	do_sample=False,
	temperature=0.2, 
	verbose=False
):
	""" Run tinyllava model inference """  
	
	# - Extract image processor from model
	logger.debug("Extracting image processor from model ...")	
	if reset_imgnorm:
		model.vision_tower._image_processor.image_mean= [0.,0.,0.]
		model.vision_tower._image_processor.image_std= [1.,1.,1.]
	
	#image_processor = model.vision_tower._image_processor
	image_processor = ImagePreprocess(model.vision_tower._image_processor, model.config)
	
	# - Extract tokenizer from model
	context_len = getattr(model.config, 'max_sequence_length', 2048)
	tokenizer = model.tokenizer
	
	# - Extract text processor from model
	text_processor = TextPreprocess(tokenizer, conv_mode)
	
	# - Set text query
	qs = query
	qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
	msg = Message()
	msg.add_message(qs)

	result= text_processor(msg.messages, mode='eval')
	input_ids= result['input_ids']
	prompt= result['prompt']
	input_ids= input_ids.unsqueeze(0).cuda()

	stop_str = text_processor.template.separator.apply()[1]
	keywords = [stop_str]
	stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

	# - Load image
	logger.debug("Loading image %s ..." % (image_path))
	image= load_img_as_pil_rgb(
		image_path,
		resize=resize, resize_size=resize_size, 
		apply_zscale=zscale, contrast=contrast,
		verbose=verbose
	)
	if image is None:
		logger.warn("Read context image %s is None, skipping inference for this ..." % (filename))
		return None
	
	if verbose:
		print("input image info: shape/min/max/mean/std")
		img_numpy= np.asarray(image)
		print(img_numpy.shape)
		print(img_numpy.min())
		print(img_numpy.max())
		print(img_numpy.mean())
		print(img_numpy.std())
	
	# - Pre-process image
	logger.debug("Preprocessing input image ...")
	image_tensor= image_processor(image)    
	image_tensor= image_tensor.unsqueeze(0).half().cuda()

	# - Generate model response
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 512
	
	with torch.inference_mode():
		output_ids= model.generate(
		input_ids,
		images=image_tensor,
		do_sample=do_sample,
		temperature=None if do_sample else temperature,
		top_p=top_p,
		num_beams=num_beams,
		pad_token_id=tokenizer.pad_token_id,
		max_new_tokens=max_new_tokens,
		use_cache=True,
		stopping_criteria=[stopping_criteria],
	)

	# - Decode output
	outputs = tokenizer.batch_decode(
		output_ids, skip_special_tokens=True
	)[0]
	outputs = outputs.strip()
	
	if verbose:
		print("outputs")
		print(outputs)
	
	if outputs.endswith(stop_str):
		outputs = outputs[: -len(stop_str)]
	outputs = outputs.strip()
	
	if verbose:
		print("outputs (parsed)")
		print(outputs)
	  
	return outputs


def run_tinyllava_model_rgz_inference(
	datalist, 
	model,
	device="cuda:0",
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	shuffle_label_options=False, nmax=-1, 
	verbose=False
):
	""" Run inference on RGZ dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	description= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	
	question_prefix= "Which of these morphological classes of radio sources do you see in the image? "
	question_subfix= "Please report only one identified class label. Report just NONE if you cannot recognize any of the above classes in the image."
	
	label2id= {
		"1C-1P": 0,
		"1C-2P": 1,
		"1C-3P": 2,
		"2C-2P": 3,
		"2C-3P": 4,
		"3C-3P": 5,
	}
	
	nclasses= len(label2id)
	class_names= list(label2id.keys())
	labels= list(label2id.values())
	
	#===========================
	#==   RUN INFERENCE
	#===========================
	# - Loop over images in dataset
	nfailed_inferences= 0
	classids= []
	classids_pred= []
	
	for idx, item in enumerate(datalist):
		# - Check stop condition
		if nmax!=-1 and idx>=nmax:
			logger.info("Stop loop condition reached (%d), as #%d entries were processed..." % (nmax, idx))
			break
	
		# - Get image info
		sname= item["sname"]
		filename= item["filepaths"][0]
		label= item["label"]
		
		# - Create question
		option_choices= class_names.copy()
		if shuffle_label_options:
			random.shuffle(option_choices)
		
		question_labels= ' \n '.join(option_choices)
		question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
		
		# - Query model
		output= run_tinyllava_model_query(
			model=model, 
			image_path=filename,
			query=question,
			reset_imgnorm=reset_imgnorm,
			resize=resize, resize_size=resize_size, 
			zscale=zscale, contrast=contrast, 
			conv_mode=conv_mode,
			do_sample=False,
			verbose=verbose
		)
		
		# - Extract predicted label
		label_pred= output.strip("\n").strip().upper()

		# - Check if label is correct
		if label_pred not in label2id:
			logger.warn("Unexpected label (%s) returned, skip this image ..." % (label_pred))
			nfailed_inferences+= 1
			continue
	
		# - Extract class ids
		classid= label2id[label]
		classid_pred= label2id[label_pred]
		classids.append(classid)
		classids_pred.append(classid_pred)	
		logger.info("image %s: GT(id=%d, label=%s), PRED(id=%d, label=%s)" % (sname, classid, label, classid_pred, label_pred))

	logger.info("#%d failed inferences" % (nfailed_inferences))

	#===========================
	#==   COMPUTE METRICS
	#===========================
	# - Compute and print metrics
	y_pred= np.array(classids_pred)
	y_true= np.array(classids)	
	metrics= multiclass_singlelabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=labels)
	print_metrics(metrics)
		
	return 0
	



