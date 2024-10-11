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
import copy

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
import torchvision.transforms as T

## LLAVA modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates

## DRAW MODULES
import matplotlib.pyplot as plt

## RADIO-LLAVA MODULES
from radio_llava.utils import *
from radio_llava.metrics import *
from radio_llava.inference_utils import *

## LOGGER
logger = logging.getLogger(__name__)

######################
##   LOAD MODEL
######################
def load_llavaov_model(model_name_or_path, is_multimodal_interleaved=False, model_name="llava_qwen", device_map="auto"):
	""" Load LLaVA One Vision model """

	# - Retrieve model name
	if model_name=="":
		logger.info("Empty model_name specified, retrieving name from model %s ..." % (model_name_or_path))
		model_name= get_model_name_from_path(model_name_or_path)

	# - Set arguments
	llava_model_args = {
		"multimodal": True,
	}
	overwrite_config = {}
	overwrite_config["image_aspect_ratio"] = "pad"
	llava_model_args["overwrite_config"] = overwrite_config

	# - Load the model
	#   NB: See https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision_Tutorials.ipynb
	logger.info("Loading model %s (name=%s) ..." % (model_name_or_path, model_name))
	if is_multimodal_interleaved:
		tokenizer, model, image_processor, max_length = load_pretrained_model(
			model_name_or_path, 
			None, 
			model_name, 
			device_map=device_map,
			**llava_model_args
		)
	else:
		tokenizer, model, image_processor, max_length = load_pretrained_model(
			model_name_or_path, 
			None, 
			model_name, 
			device_map=device_map
		)
		
	#model.generation_config.pad_token_id = model.generation_config.eos_token_id
	model.eval()

	return model, tokenizer, image_processor

####################################
##   INFERENCE UTILS
####################################
def run_llavaov_model_query(
	model,
	tokenizer,
	image_processor, 
	image, 
	query,
	do_sample=False,
	temperature=0.2,
	conv_template="qwen_2", 
	verbose=False
):
	""" Run llava one vision model inference """  
	
	
	# - Process image
	image_tensor = process_images([image], image_processor, model.config)
	image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]

	# - Create prompt 
	question = DEFAULT_IMAGE_TOKEN + "\n" + query
	conv = copy.deepcopy(conv_templates[conv_template])
	conv.append_message(conv.roles[0], question)
	conv.append_message(conv.roles[1], None)
	prompt_question = conv.get_prompt()
	
	if verbose:
		print("prompt_question")
		print(prompt_question)

	# - Create model inputs
	input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
	image_sizes = [image.size]

	# - Generate model response
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 4096
	
	output = model.generate(
		input_ids,
		images=image_tensor,
		image_sizes=image_sizes,
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens,
		#use_cache=True,
	)

	#output_parsed = tokenizer.batch_decode(output, skip_special_tokens=True)
	output_parsed= tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	#output_parsed_list= output_parsed.split("assistant")
	
	if verbose:
		print("output")
		print(output)

		print("output_parsed")
		print(output_parsed)
		
		#print("output_parsed (split assistant)")
		#print(output_parsed_list)
			
	# - Extract predicted label
	#response= output_parsed_list[-1].strip("\n").strip()
	#response= output_parsed
	response= output_parsed.strip("\n").strip()
	
	return response
	


def run_llavaov_model_context_query(
	model,
	tokenizer,
	image_processor, 
	image, 
	query,
	images_context,
	conversations_context,
	do_sample=False,
	temperature=0.2,
	conv_template="qwen_2",
	verbose=False
):
	""" Run llava one vision model inference """  

	# - Check context info
	if not images_context or not conversations_context:
		logger.error("Empty list given for either context images, queries or responses!")
		return None
		
	# - Process images
	images= images_context.copy()
	images.append(image)
	image_tensors = process_images(images, image_processor, model.config)
	image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
	image_sizes = [image.size for image in images]
	
	# - Create context prompts
	prompts= []
	for item in conversations_context:
		conv = copy.deepcopy(conv_templates[conv_template])
		query_context= item['question']
		response_context= item['response']
		question_context= DEFAULT_IMAGE_TOKEN + "\n" + query_context
		conv.append_message(conv.roles[0], question_context)
		conv.append_message(conv.roles[1], response_context)
		prompt_context = conv.get_prompt()
		prompts.append(prompt_context)
		
	# - Create question prompts	
	question = DEFAULT_IMAGE_TOKEN + "\n" + query

	conv = copy.deepcopy(conv_templates[conv_template])
	conv.append_message(conv.roles[0], question)
	conv.append_message(conv.roles[1], None)	
	prompt = conv.get_prompt()
	prompts.append(prompt)
	
	print("--> prompts")
	print(prompts)
	
	# - Create inputs
	input_ids = tokenizer_image_token(prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
	
	print("--> inputs")
	print(inputs)

	# - Generate model response
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 4096
	
	output = model.generate(
		input_ids,
		images=image_tensor,
		image_sizes=image_sizes,
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens,
		#use_cache=True,
	)
	
	print("--> output")
	print(output)
	
	output_parsed= tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	output_parsed= tokenizer.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	
	print("--> output_parsed")
	print(output_parsed)
	
	return output_parsed
	
	
	

def run_llavaov_model_inference(
	datalist, 
	model, tokenizer, image_processor, 
	task_info, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	add_options=False, shuffle_options=False,
	nmax=-1,
	nmax_context=-1,
	conv_template="qwen_1_5",
	verbose=False
):
	""" Run LLaVA One Vision inference on radio image dataset """

	#====================================
	#==   GET TASK INFO
	#====================================
	# - Get query info
	description= task_info["description"]
	question_prefix= task_info["question_prefix"]
	question_subfix= task_info["question_subfix"]
	
	# - Get class info
	classification_mode= task_info["classification_mode"]
	label_modifier_fcn= task_info["label_modifier_fcn"]
	label2id= task_info["label2id"]
	nclasses= len(label2id)
	class_names= list(label2id.keys())
	labels= list(label2id.values())

	class_options= class_names
	if "class_options" in task_info:
		class_options= task_info["class_options"]

	#====================================
	#==   CREATE CONTEXT CONVERSATIONS
	#====================================
	# - Create context conversation (if datalist_context if given)
	conversations_context= []
	images_context= []
	
	if datalist_context is not None:
		for idx, item in enumerate(datalist_context):
			# - Check stop condition
			if nmax_context!=-1 and idx>=nmax_context:
				logger.info("Stop loop condition reached (%d) for context data, as #%d entries were processed..." % (nmax_context, idx))
				break
			
			# - Get image info
			sname= item["sname"]
			filename= item["filepaths"][0]
			label= item["label"]
			
			
			# - Read image into PIL
			image= load_img_as_pil_rgb(
				filename,
				resize=resize, resize_size=resize_size, 
				apply_zscale=zscale, contrast=contrast,
				verbose=verbose
			)
			if image is None:
				logger.warn("Read context image %s is None, skipping inference for this ..." % (filename))
				continue
			
			images_context.append(image)
			
			# - Create question
			if add_options:
				option_choices= class_options.copy()
				question_labels= ' \n '.join(option_choices)
				if idx==0:
					question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
				else:
					question= question_prefix + ' \n ' + question_labels + question_subfix
			else:
				question= description + ' \n' + question_prefix + ' \n ' + question_subfix
		
			
			# - Set assistant response to true label
			response= format_context_model_response(label, classification_mode, label_modifier_fcn)
			#response= label
			
			# - Create conversation
			conversation= {
				"question": question,
				"response": response
			}
			
			#conversation = [
			#	{
			#		"role": "user",
			#		"content": [
			#			{"type": "image"},
			#			{"type": "text", "text": question},
			#		],
    	#	},
    	#	{
			#		"role": "assistant",
			#		"content": [
			#			{"type": "text", "text": response},
			#		],
			#	},
			#]
			conversations_context.extend(conversation)

	#===========================
	#==   RUN INFERENCE
	#===========================
	ninferences_unexpected= 0
	ninferences_failed= 0
	class_ids= []
	class_ids_pred= []
	
	for idx, item in enumerate(datalist):
		
		# - Check stop condition
		if nmax!=-1 and idx>=nmax:
			logger.info("Stop loop condition reached (%d), as #%d entries were processed..." % (nmax, idx))
			break
			
		# - Get image info
		sname= item["sname"]
		filename= item["filepaths"][0]
		label= item["label"]
		
		# - Load into PIL
		image= load_img_as_pil_rgb(
			filename,
			resize=resize, resize_size=resize_size, 
			apply_zscale=zscale, contrast=contrast,
			verbose=verbose
		)
		if image is None:
			logger.warn("Read context image %s is None, skipping inference for this ..." % (filename))
			ninferences_failed+= 1
			continue

		# - Create question
		if add_options:
			option_choices= class_options.copy()
			if shuffle_options:
				random.shuffle(option_choices)
		
			question_labels= ' \n '.join(option_choices)
			if conversations_context:
				question= question_prefix + ' \n ' + question_labels + question_subfix
			else:
				question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
		else:
			if conversations_context:
				question= question_prefix + ' \n ' + question_subfix
			else:
				question= description + ' \n' + question_prefix + ' \n ' + question_subfix

		# - Run inference with or without context
		if conversations_context:
			output= run_llavaov_model_context_query(
				model, tokenizer, image_processor, 
				image, 
				question,
				images_context,
				conversations_context,
				do_sample=False,
				temperature=None,
				conv_template=conv_template,
				verbose=verbose
			)
		else:
			output= run_llavaov_model_query(
				model, tokenizer, image_processor, 
				image, 
				question,
				do_sample=False,
				temperature=None,
				conv_template=conv_template,
				verbose=verbose
			)

		if output is None:
			logger.warn("Failed inference for image %s, skipping ..." % (filename))
			ninferences_failed+= 1
			continue

		#########################
		##   PROCESS OUTPUT
		#########################
		# - Extract class ids
		res= process_model_output(output, label, label2id, classification_mode, label_modifier_fcn)
		if res is None:
			logger.warn("Unexpected label prediction found, skip this image ...")
			ninferences_unexpected+= 1
			continue

		classid= res[0]
		classid_pred= res[1]
		label= res[2]
		label_pred= res[3]
		
		class_ids.append(classid)
		class_ids_pred.append(classid_pred)	
		logger.info("image %s: GT(id=%s, label=%s), PRED(id=%s, label=%s)" % (sname, str(classid), str(label), str(classid_pred), str(label_pred)))

	logger.info("#%d failed inferences" % (ninferences_failed))
	logger.info("#%d unexpected inferences" % (ninferences_unexpected))

	#===========================
	#==   COMPUTE METRICS
	#===========================
	# - Compute and print metrics
	if not class_ids_pred or not class_ids:
		logger.warn("class_ids or class_ids_pred are empty, skip metric calculation!")
		return 0
	
	y_pred= np.array(class_ids_pred)
	y_true= np.array(class_ids)

	if classification_mode=="multiclass_multilabel":
		metrics= multiclass_multilabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=labels)
		print_metrics(metrics)
		
	elif classification_mode=="multiclass_singlelabel":
		metrics= multiclass_singlelabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=labels)
		print_metrics(metrics)
		
	else:
		logger.error("Invalid/unknown classification mode (%s) given!" % (classification_mode))
		return -1
		
	return 0

#############################################
###       INFERENCE TASKS
#############################################
def run_llavaov_model_rgz_inference(
	datalist, 
	model, tokenizer, image_processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1,
	nmax_context=-1,
	add_task_description=False,
	conv_template="qwen_1_5",
	verbose=False
):
	""" Run LLaVA One Vision inference on RGZ dataset """

	#===========================
	#==   INIT TASK
	#===========================
	
	# - Define message
	if add_task_description:
		description= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. \n An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	else:
		description= ""
		
	question_prefix= "Which of these morphological classes of radio sources do you see in the image? "
	question_subfix= "Please report only the identified class label, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	
	label2id= {
		"1C-1P": 0,
		"1C-2P": 1,
		"1C-3P": 2,
		"2C-2P": 3,
		"2C-3P": 4,
		"3C-3P": 5,
	}
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": None,
		"label2id": label2id
	}
	
	#=============================
	#==   RUN TASK
	#=============================
	return run_llavaov_model_inference(
		datalist, 
		model, tokenizer, image_processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		conv_template=conv_template,
		verbose=verbose
	)
		

def run_llavaov_model_smorph_inference(
	datalist, 
	model, tokenizer, image_processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	conv_template="qwen_1_5",
	verbose=False
):
	""" Run LLaVA One Vision inference on radio image dataset """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if add_task_description:
		description= "Consider these morphological classes of radio astronomical sources, defined as follows: \n EXTENDED: This class comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components. Typical examples are extended radio galaxies formed by a single elongated island or by multiple islands, hosting the galaxy core and lobe structures; \n DIFFUSE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology; \n DIFFUSE-LARGE: large-scale (e.g. larger than few arcminutes and covering a large portion of the image) diffuse object with irregular shape. \n An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	else:
		description= ""
		
	question_prefix= "Which of these morphological classes of radio sources do you see in the image? "
	question_subfix= "Please report the identified class labels separated by commas, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	
	label2id= {
		"NONE": 0,
		"EXTENDED": 1,
		"DIFFUSE": 2,
		"DIFFUSE-LARGE": 3
	}
	
	class_options= ["EXTENDED","DIFFUSE","DIFFUSE-LARGE"]
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_multilabel",
		"label_modifier_fcn": filter_smorph_label,
		"label2id": label2id,
		"class_options": class_options,
	}
	
	#=============================
	#==   RUN TASK
	#=============================
	return run_llavaov_model_inference(
		datalist, 
		model, tokenizer, image_processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		conv_template=conv_template,
		verbose=verbose
	)	
	



