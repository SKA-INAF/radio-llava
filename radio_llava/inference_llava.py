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
import torchvision.transforms as T
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

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
def load_llavaov_model(model_name_or_path, device="cuda"):
	""" Load LLaVA One Vision model """

	# - Load the model in half-precision
	logger.info("Loading model %s ..." % (model_name_or_path))
	model = LlavaOnevisionForConditionalGeneration.from_pretrained(
		model_name_or_path, 
		torch_dtype=torch.float16, 
		device_map="auto"
	)
	
	model.generation_config.pad_token_id = model.generation_config.eos_token_id
	model.eval()

	# - Load processor
	logger.info("Loading processor for model %s ..." % (model_name_or_path))
	processor = AutoProcessor.from_pretrained(model_name_or_path)
	
	return model, processor	
	

#############################
##   INFERENCE UTILS
#############################
def run_llavaov_model_query(
	model,
	processor, 
	image, 
	query,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	do_sample=False,
	temperature=0.2, 
	verbose=False
):
	""" Run llava one vision model inference """  
	
	# - Create conversation (eventually adding also context conversations)
	conversation= [
		{
			"role": "user",
			"content": [
				{"type": "image"},
				{"type": "text", "text": query},
			],
		},
	]
		
	# - Create prompt 
	prompt= processor.apply_chat_template(conversation, add_generation_prompt=True)
	
	# - Create model inputs (eventually combining context and inference prompts)
	inputs= processor(image, prompt, return_tensors="pt").to(model.device, torch.float16)
	
	if verbose:
		print("conversations")
		print(json.dumps(conversation, indent=2))
		print("inputs")
		print(inputs)
		print("inputs.pixel_values")
		print(inputs['pixel_values'].shape)

	# - Generate model response
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 512
	
	output = model.generate(
		**inputs, 
		do_sample=do_sample,
		temperature=None if do_sample else temperature,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens,
		use_cache=True,
	)
		
	# - Decode response
	output_parsed= processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	output_parsed_list= output_parsed.split("assistant")
	
	if verbose:
		print("output")
		print(output)

		print("output_parsed")
		print(output_parsed)
			
		print("output_parsed (split assistant)")
		print(output_parsed_list)
		
	# - Extract predicted label
	response= output_parsed_list[-1].strip("\n").strip()
		
	return response


def run_llavaov_model_context_query(
	model,
	processor, 
	image, 
	query,
	images_context,
	conversations_context,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	do_sample=False,
	temperature=0.2, 
	verbose=False
):
	""" Run llava one vision model inference """  

	# - Check context info
	if not images_context or not conversations_context:
		logger.error("Empty list given for either context images, queries or responses!")
		return None
	
	# - Create conversation and add after context conversations)
	conversation= [
		{
			"role": "user",
			"content": [
				{"type": "image"},
				{"type": "text", "text": query},
			],
    },
	]
	conversations= conversations_context.copy()
	conversations.extend(conversation)
		
	# - Create prompt 
	prompts= processor.apply_chat_template(conversations, add_generation_prompt=True)
		
	# - Create model inputs (combining context and inference prompts)
	images= images_context.copy()
	images.append(image)
		
	inputs = processor(images, prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
		
	if verbose:
		print("conversations")
		print(json.dumps(conversations, indent=2))
		print("inputs")
		print(inputs)
		print("inputs.pixel_values")
		print(inputs['pixel_values'].shape)

	# - Generate model response
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 512
	
	output = model.generate(
		**inputs, 
		do_sample=do_sample,
		temperature=None if do_sample else temperature,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens,
		use_cache=True,
	)
		
	# - Decode response
	output_parsed= processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	output_parsed_list= output_parsed.split("assistant")

	if verbose:
		print("output")
		print(output)

		print("output_parsed")
		print(output_parsed)
		
		print("output_parsed (split assistant)")
		print(output_parsed_list)
		
	# - Extract predicted label
	response= output_parsed_list[-1].strip("\n").strip()

	return response
	


def run_llavaov_model_inference(
	datalist, 
	model, processor, 
	task_info, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_label_options=False, 
	nmax=-1,
	nmax_context=-1,
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
			if nmax_context!=-1 and idx>=nmax:
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
			option_choices= class_options.copy()
			question_labels= ' \n '.join(option_choices)
			if idx==0:
				question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
			else:
				question= question_prefix + ' \n ' + question_labels + question_subfix
		
			# - Set assistant response to true label
			response= format_context_model_response(label, classification_mode, label_modifier_fcn)
			#response= label
			
			# - Create conversation
			conversation = [
				{
					"role": "user",
					"content": [
						{"type": "image"},
						{"type": "text", "text": question},
					],
    		},
    		{
					"role": "assistant",
					"content": [
						{"type": "text", "text": response},
					],
				},
			]
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
		option_choices= class_options.copy()
		if shuffle_label_options:
			random.shuffle(option_choices)
		
		question_labels= ' \n '.join(option_choices)
		if conversations_context:
			question= question_prefix + ' \n ' + question_labels + question_subfix
		else:
			question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix

		# - Run inference with or without context
		if conversations_context:
			output= run_llavaov_model_context_query(
				model, processor, 
				image, 
				question,
				images_context,
				conversations_context,
				resize=resize, resize_size=resize_size, 
				zscale=zscale, contrast=contrast, 
				do_sample=False,
				temperature=None,
				verbose=verbose
			)
		else:
			output= run_llavaov_model_query(
				model, processor, 
				image, 
				question,
				resize=resize, resize_size=resize_size, 
				zscale=zscale, contrast=contrast, 
				do_sample=False,
				temperature=None,
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
	model, processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_label_options=False, 
	nmax=-1,
	nmax_context=-1,
	add_task_description=False,
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
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		shuffle_label_options=shuffle_label_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)
		

def run_llavaov_model_smorph_inference(
	datalist, 
	model, processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_label_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
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
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		shuffle_label_options=shuffle_label_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	



