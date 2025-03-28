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
from radio_llava import logger

######################
##   LOAD MODEL
######################
def load_llavaov_model(model_name_or_path, is_multimodal_interleaved=False, model_name="llava_qwen", model_base=None, device_map="auto"):
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
			model_base, 
			model_name, 
			device_map=device_map,
			**llava_model_args
		)
	else:
		tokenizer, model, image_processor, max_length = load_pretrained_model(
			model_name_or_path, 
			model_base, 
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
	max_new_tokens=4096,
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
	conv.system= '<|im_start|>system\nYou are an AI assistant specialized in radio astronomical topics.'
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
	max_new_tokens=4096,
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
	
	# - Create prompts (first add context prompt then user question)
	conv = copy.deepcopy(conv_templates[conv_template])
	for item in conversations_context:
		query_context= item['question']
		response_context= item['response']
		question_context= DEFAULT_IMAGE_TOKEN + "\n" + query_context
		conv.append_message(conv.roles[0], question_context)
		conv.append_message(conv.roles[1], response_context)
	
	question = DEFAULT_IMAGE_TOKEN + "\n" + query
	conv.append_message(conv.roles[0], question)
	conv.append_message(conv.roles[1], None)	

	prompt = conv.get_prompt()
	
	if verbose:
		print("--> conversations_context")
		print(conversations_context)

		print("--> prompt")
		print(prompt)
	
	# - Create inputs
	input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
	
	if verbose:
		print("--> inputs")
		print(input_ids)
		print(input_ids.shape)

	# - Generate model response
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	
	output = model.generate(
		input_ids,
		images=image_tensors,
		image_sizes=image_sizes,
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens,
		#use_cache=True,
	)
	
	output_parsed= tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	response= output_parsed.strip("\n").strip()
	
	if verbose:
		print("--> output")
		print(output)
	
		print("--> output_parsed")
		print(output_parsed)
	
	
	return response
	

def run_llavaov_model_inference_on_image(
	image_path,
	model, tokenizer, image_processor,
	prompt,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	do_sample=False,
	temperature=0.2,
	max_new_tokens=4096,
	conv_template="qwen_2",
	verbose=False
):
	""" Run LLaVA One Vision inference on radio image """

	# - Load image as PIL
	image= load_img_as_pil_rgb(
		image_path,
		resize=resize, resize_size=resize_size, 
		apply_zscale=zscale, contrast=contrast,
		verbose=verbose
	)
	if image is None:
		logger.warn("Failed to read image from file %s!" % (image_path))
		return None

	# - Run inference without context
	output= run_llavaov_model_query(
		model, tokenizer, image_processor, 
		image, 
		prompt,
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		max_new_tokens=max_new_tokens,
		conv_template=conv_template,
		verbose=verbose
	)
				
	if output is None:
		logger.warn("Failed inference for image %s!" % (image_path))

	return output		


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
	do_sample=False,
	temperature=0.2,
	max_new_tokens=4096,
	conv_template="qwen_2",
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
				logger.warn("Read context image %s is None, skipping inference for this image ..." % (filename))
				continue
			
			images_context.append(image)
			
			# - Create question
			if add_options:
				option_choices= class_options.copy()
				question_labels= ' \n '.join(option_choices)
				if idx==0:
					question= description + ' \n' + question_prefix + ' \n ' + question_labels + ' \n ' + question_subfix
				else:
					question= question_prefix + ' \n ' + question_labels + ' \n ' + question_subfix
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
			
			conversations_context.append(conversation)

	#===========================
	#==   RUN INFERENCE
	#===========================
	ninferences_unexpected= 0
	ninferences_failed= 0
	class_ids= []
	class_ids_pred= []
	n_max_retries= 3
	
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
			question= description + ' \n' + question_prefix + ' \n ' + question_labels + ' \n ' + question_subfix
		else:
			question= description + ' \n' + question_prefix + ' \n ' + question_subfix
			
		#question_retry= "The format of your response does not comply with the requested instructions, please answer again to the following request and strictly follow the given instructions. \n" + question
		question_retry= "The format of your response does not comply with the requested instructions, please answer again to the question and strictly follow the instructions: \n" + question_subfix
		
		skip_inference= False
		n_retries= 0
		
		while n_retries<=n_max_retries:
			#########################
			##   RUN INFERENCE
			#########################
			# - Run inference with or without context
			if n_retries==0:
				question_curr= question
			else:
				question_curr= question_retry
				
			#print("question: ", question_curr)
						
			if conversations_context:
				output= run_llavaov_model_context_query(
					model, tokenizer, image_processor, 
					image, 
					question_curr,
					images_context,
					conversations_context,
					do_sample=do_sample,
					temperature=temperature if do_sample else None,
					max_new_tokens=max_new_tokens,
					conv_template=conv_template,
					verbose=verbose
				)
			else:
				output= run_llavaov_model_query(
					model, tokenizer, image_processor, 
					image, 
					question_curr,
					do_sample=do_sample,
					temperature=temperature if do_sample else None,
					max_new_tokens=max_new_tokens,
					conv_template=conv_template,
					verbose=verbose
				)
				
			if output is None:
				logger.warn("Failed inference for image %s, skipping ..." % (filename))
				skip_inference= True
				break
				#ninferences_failed+= 1
				#continue

			#########################
			##   PROCESS OUTPUT
			#########################
			# - Extract class ids
			res= process_model_output(output, label, label2id, classification_mode, label_modifier_fcn)
			#print("output:", output)
			#print("res: ", res)
			#print("type(res)", type(res))
			
			if res is None:
				#print("res is None!")
				if n_retries>=n_max_retries:
					#print("Unexpected label prediction obtained for image, giving up and skipping image")
					logger.warn("Unexpected label prediction obtained for image %s, giving up and skipping image ..." % (filename))
					skip_inference= True
					break
				else:
					n_retries+= 1
					#print("Unexpected label prediction obtained for image, trying again ...")
					logger.warn("Unexpected label prediction obtained for image %s, trying again (#nretry=%d) ..." % (filename, n_retries))
					continue
			else:
				logger.info("Correct label prediction obtained for image %s, computing class ids ..." % (filename))	
				break
							
		# - Check if inference has to be skipped for this image
		if skip_inference or res is None:
			ninferences_failed+= 1
			continue

		# - Post process results
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA One Vision inference on RGZ dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		context= "### Context: Consider these morphological classes of radio astronomical sources: \n 1C-1P: single-island sources having only one flux intensity peak; \n 1C-2C: single-island sources having two flux intensity peaks; \n 1C-3P: single-island sources having three flux intensity peaks; \n 2C-2P: sources consisting of two separated islands, each hosting a single flux intensity peak; \n 2C-3P: sources consisting of two separated islands, one containing a single peak of flux intensity and the other exhibiting two distinct intensity peaks; \n 3C-3P: sources consisting of three separated islands, each hosting a single flux intensity peak. \n An island is a group of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
		context+= "\n"

		description= ""
		if add_task_description: 
			description= context
		
		question_prefix= "### Question: Which of these morphological classes of radio sources do you see in the image? "
		if add_task_description:
			if datalist_context is None:
				question_subfix= "Answer the question using the provided context. "
			else:
				question_subfix= "Answer the question using the provided context and examples. "
		else:
			if datalist_context is None:
				question_subfix= ""
			else:
				question_subfix= "Answer the question using the provided examples. "
	
		question_subfix+= "Report only the identified class label, without any additional explanation text."
		
	elif prompt_version=="v2":
		description= "You are given a radio astronomical input image. Answer to the question below, strictly following the provided instructions. \n"
	
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level. The terms 'island' and 'component' are sometimes used interchangeably, but an 'island' is defined purely by pixel connectivity above a noise threshold, and may contain one or multiple source components. Example: A radio galaxy with extended lobes may appear as one large source island, encompassing multiple structures (core, jets, lobes).\n"
			context+= "- SOURCE COMPONENT: an individual emission structure within a radio astronomical image that is associated with a physically coherent entity or a substructure of a larger source. In source extraction, a single astronomical source may be represented by one or multiple components, depending on its morphology and how the extraction algorithm segments emission regions. These components may correspond to compact sources, such as individual galaxies or quasars, or extended sources, such as lobes of radio galaxies or supernova remnants. Example: A radio galaxy with a central core and two extended lobes may be decomposed into three source components—one for the core and one for each lobe. A single source island may contain multiple source components (e.g., a double-lobed radio galaxy).\n"
			context+= "- SOURCE INTENSITY PEAK: the location of the maximum brightness (or flux density) within a detected source component in a radio astronomical image. It represents the highest observed intensity value in the 2D brightness distribution of the source, typically measured in Jy/beam (Jansky per beam). The peak intensity is crucial for identifying the position of the source, as well as for distinguishing between compact and extended emission structures. Example: For a compact radio source, such as a quasar, the intensity peak often coincides with the centroid of the source. In contrast, for an extended radio galaxy, the peak intensity may be located in the core or along the brightest part of the radio lobes. \n"
			context+= "- 1C-1P SOURCE: morphological class of single-island radio sources having only one flux intensity peak. \n"
			context+= "- 1C-2P SOURCE: morphological class of single-island radio sources having two flux intensity peaks. \n"
			context+= "- 1C-3P SOURCE: morphological class of single-island radio sources having three flux intensity peaks. \n"
			context+= "- 2C-2P SOURCE: morphological class of radio sources formed by two disjoint islands, each hosting a single flux intensity peak. \n"
			context+= "- 2C-3P SOURCE: morphological class of radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks. \n"
			context+= "- 3C-3P SOURCE: morphological class of radio sources formed by three disjoint islands, each hosting a single flux intensity peak."
			description+= context
		
		question_prefix= "\n ## Question: Which of these morphological classes of radio sources do you see in the image? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely reporting only the identified source class label taken from these possible choices: 1C-1P, 1C-2P, 1C-3P, 2C-2P, 2C-3P, 3C-3P. DO NOT ADD ANY EXPLANATION TEXT. \n"
		#question_subfix+= "- Answer just NONE if you cannot recognize any of the above classes in the image. \n"
		#question_subfix+= "- Answer directly from the given context. \n"
		#question_subfix+= "- Your response must be just the identified class label taken from these possible choices: 1C-1P, 1C-2P, 1C-3P, 2C-2P, 2C-3P, 3C-3P \n"
		#question_subfix+= "- Answer NONE if you cannot recognize any of the above classes in the image. \n"
		#question_subfix+= "- If the context doesn't contain relevant information, respond with: \"I don't have enough information to answer this question.\" \n"
		#question_subfix+= "- Do not add explanation or description texts to the response. \n"
		#question_subfix+= "\n"

	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return	
	
	
	# - Define class options
	class_options= ["1C-1P", "1C-2P", "1C-3P", "2C-2P", "2C-3P", "3C-3P"]
	
	label2id= {
		"NONE": 0,
		"1C-1P": 1,
		"1C-2P": 2,
		"1C-3P": 3,
		"2C-2P": 4,
		"2C-3P": 5,
		"3C-3P": 6,
	}
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": None,
		"label2id": label2id,
		"class_options": class_options
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA One Vision inference on radio image dataset """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		context= "### Context: Consider these morphological classes of radio astronomical sources, defined as follows: \n EXTENDED: This class comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components. Typical examples are extended radio galaxies formed by a single elongated island or by multiple islands, hosting the galaxy core and lobe structures; \n DIFFUSE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology; \n DIFFUSE-LARGE: large-scale (e.g. larger than few arcminutes and covering a large portion of the image) diffuse object with irregular shape. \n An island is a group of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level."
	
		description= ""
		if add_task_description:
			description= context
	
		question_prefix= "### Question: Which of these morphological classes of radio sources do you see in the image? "
	
		if add_task_description:
			if datalist_context is None:
				question_subfix= "Answer the question using the provided context. "
			else:
				question_subfix= "Answer the question using the provided context and examples. "
		else:
			if datalist_context is None:
				question_subfix= ""
			else:
				question_subfix= "Answer the question using the provided examples. "
			
		question_subfix+= "Report the identified class labels separated by commas, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	
	elif prompt_version=="v2":
		description= "You are given a radio astronomical input image. Answer to the question below, strictly following the provided instructions. \n"
	
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level. \n"
			context+= "- COMPACT SOURCE: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape. \n"
			context+= "- EXTENDED SOURCE: This morphological class of radio sources comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components. \n"
			context+= "- DIFFUSE SOURCE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology. \n"
			context+= "- DIFFUSE-LARGE SOURCE: large-scale (e.g. larger than few arcminutes and covering a large portion of the image) diffuse object with irregular shape."
			description+= context
		
		question_prefix= "\n ## Question: Which of these morphological classes of radio sources do you see in the image? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely reporting only the identified source class labels {'EXTENDED','DIFFUSE','DIFFUSE-LARGE'} separated by commas. DO NOT ADD ANY EXPLANATION TEXT. \n"
		question_subfix+= "- Answer just NONE if you cannot recognize any of the above classes in the image. \n"
	
	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return
	
	# - Define label options
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
	

def run_llavaov_model_galaxy_inference(
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA inference on radio image dataset (galaxy detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		description= ""
		question_prefix= "Do you see any likely radio galaxy with an extended morphology in the image? "
		question_subfix= "Answer concisely: Yes or No."
	
	elif prompt_version=="v2":
		description= "You are given a radio astronomical input image. Answer to the question below, strictly following the provided instructions. \n"
	
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- RADIO GALAXY: a type of active galaxy that emits an exceptionally large amount of radio waves, often extending beyond its visible structure. These galaxies host an active galactic nucleus (AGN), powered by a supermassive black hole (SMBH) at their center, which fuels the production of powerful radio jets and lobes. "
			description+= context
		
		question_prefix= "\n ## Question: Do you see any likely radio galaxy with an extended morphology in the image? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely: Yes or No. \n"
		
	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return
	
	# - Define class options
	label2id= {
		"NO": 0,
		"YES": 1,
	}
	
	class_options= None
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": filter_galaxy_label,
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
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		conv_template=conv_template,
		verbose=verbose
	)	
	
def run_llavaov_model_artefact_inference(
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA inference on radio image dataset (artefact detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		description= ""
		question_prefix= "Do you see any imaging artefact with a ring pattern around bright sources in the image? "
		question_subfix= "Answer concisely: Yes or No."
	
	elif prompt_version=="v2":
		description= "You are given a radio astronomical input image. Answer to the question below, strictly following the provided instructions. \n"
	
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- ARTEFACT: noise pattern with a ring-like or elongated morphology, typically found around bright compact sources in radio images. They are often mistaken for real radio sources but are actually spurious. These patterns arise from imperfections in the imaging processing stage of radio data."
			description+= context
		
		question_prefix= "\n ## Question: Do you see any imaging artefact with a ring pattern around bright sources in the image? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely: Yes or No. \n"
		
	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return
		
	# - Define class options
	label2id= {
		"NO": 0,
		"YES": 1,
	}
	
	class_options= None
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": filter_artefact_label,
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
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		conv_template=conv_template,
		verbose=verbose
	)	

def run_llavaov_model_anomaly_inference(
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA inference on radio image dataset (anomaly detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		context= "### Context: Consider this radio image peculiarity classes, defined as follows: \n ORDINARY: image containing only point-like or slightly-resolved compact radio sources superimposed over the sky background or imaging artefact patterns; \n COMPLEX: image containing one or more radio sources with extended or diffuse morphology; \n PECULIAR: image containing one or more radio sources with anomalous or peculiar extended morphology, often having diffuse edges, complex irregular shapes, covering a large portion of the image.\n"
	
		description= ""
		if add_task_description:
			description= context
		
		question_prefix= "### Question: Can you identify which peculiarity class the presented image belongs to? "
	
		if add_task_description:
			if datalist_context is None:
				question_subfix= "Answer the question using the provided context. "
			else:
				question_subfix= "Answer the question using the provided context and examples. "
		else:
			if datalist_context is None:
				question_subfix= ""
			else:
				question_subfix= "Answer the question using the provided examples. "
			
		question_subfix+= "Report only the identified class label, without any additional explanation text."
	
	elif prompt_version=="v2":
		description= "You are given a radio astronomical input image. Answer to the question below, strictly following the provided instructions. \n"
		
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level.\n"
			context+= "- COMPACT SOURCE: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape. \n"
			context+= "- EXTENDED SOURCE: This morphological class of radio sources comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components. \n"
			context+= "- DIFFUSE SOURCE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology \n"
			context+= "- ORDINARY IMAGE: image containing only point-like or slightly-resolved compact radio sources superimposed over the sky background or imaging artefact patterns. \n"
			context+= "- COMPLEX IMAGE: image containing one or more radio sources with extended or diffuse morphology. \n"
			context+= "- PECULIAR IMAGE: image containing one or more radio sources with anomalous or peculiar extended morphology, often having diffuse edges, complex irregular shapes, covering a large portion of the image."
			description+= context
		
		question_prefix= "\n ## Question: Can you identify which peculiarity class the presented image belongs to? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely reporting only the identified source class label taken from these possible choices: ORDINARY, COMPLEX, PECULIAR. DO NOT ADD ANY EXPLANATION TEXT. \n"
		
	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return
	
	# - Define class options
	label2id= {
		"ORDINARY": 0,
		"COMPLEX": 1,
		"PECULIAR": 2,
	}
	
	class_options= ["ORDINARY","COMPLEX","PECULIAR"]
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": filter_anomaly_label,
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
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		conv_template=conv_template,
		verbose=verbose
	)	
	


def run_llavaov_model_mirabest_inference(
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA One Vision inference on Mirabest dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		context= "### Context: Consider these morphological classes of radio galaxies: \n FR-I: radio-loud galaxies characterized by a jet-dominated structure where the radio emissions are strongest close to the galaxy's center and diminish with distance from the core; \n FR-II: radio-loud galaxies characterized by a edge-brightened radio structure, where the radio emissions are more prominent in lobes located far from the galaxy's core, with hotspots at the ends of powerful, well-collimated jets. "
		context+= "\n"
	
		description= ""
		if add_task_description: 
			description= context
		
		question_prefix= "### Question: Which of these morphological classes of radio galaxy do you see in the image? "
		if add_task_description:
			if datalist_context is None:
				question_subfix= "Answer the question using the provided context. "
			else:
				question_subfix= "Answer the question using the provided context and examples. "
		else:
			if datalist_context is None:
				question_subfix= ""
			else:
				question_subfix= "Answer the question using the provided examples. "
			
		question_subfix+= "Report only the identified class label, without any additional explanation text."
	
	elif prompt_version=="v2":
		description= "You are given a radio astronomical input image. Answer to the question below, strictly following the provided instructions. \n"
		
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- RADIO GALAXY: a type of active galaxy that emits an exceptionally large amount of radio waves, often extending beyond its visible structure. These galaxies host an active galactic nucleus (AGN), powered by a supermassive black hole (SMBH) at their center, which fuels the production of powerful radio jets and lobes. \n"
			context+= "- FR-I RADIO GALAXY: radio-loud galaxies characterized by a jet-dominated structure where the radio emissions are strongest close to the galaxy's center and diminish with distance from the core. \n"
			context+= "- FR-II RADIO GALAXY: radio-loud galaxies characterized by a edge-brightened radio structure, where the radio emissions are more prominent in lobes located far from the galaxy's core, with hotspots at the ends of powerful, well-collimated jets."
			description+= context
		
		question_prefix= "\n ## Question: Which of these morphological classes of radio galaxy do you see in the image? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely reporting only the identified source class label taken from these possible choices: FR-I, FR-II. DO NOT ADD ANY EXPLANATION TEXT. \n"
		
	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return
	
	# - Define class options
	class_options= ["FR-I", "FR-II"]
	
	label2id= {
		"NONE": 0,
		"FR-I": 1,
		"FR-II": 2
	}
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": None,
		"label2id": label2id,
		"class_options": class_options
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
	
	
	
def run_llavaov_model_gmnist_inference(
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
	conv_template="qwen_2",
	prompt_version="v1",
	verbose=False
):
	""" Run LLaVA One Vision inference on Galaxy MNIST dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if prompt_version=="v1":
		context= "### Context: Consider these morphological classes of optical galaxies: \n SMOOTH_ROUND: smooth and round galaxy. Should not have signs of spires; \n SMOOTH_CIGAR: smooth and cigar-shaped galaxy, looks like being seen edge on. This should not have signs of spires of a spiral galaxy; \n EDGE_ON_DISK: edge-on-disk/spiral galaxy. This disk galaxy should have signs of spires, as seen from an edge-on perspective; \n UNBARRED_SPIRAL: unbarred spiral galaxy. Has signs of a disk and/or spires. \n Note that categories SMOOTH_CIGAR and EDGE_ON_DISK classes tend to be very similar to each other. To categorize them, ask yourself the following question: Is this galaxy very smooth, maybe with a small bulge? Then it belongs to class SMOOTH_CIGAR. Does it have irregularities/signs of structure? Then it belongs to class EDGE_ON_DISK."
		context+= "\n"
	
		description= ""
		if add_task_description: 
			description= context
		
		question_prefix= "### Question: Which of these morphological classes of optical galaxy do you see in the image? "
		if add_task_description:
			if datalist_context is None:
				question_subfix= "Answer the question using the provided context. "
			else:
				question_subfix= "Answer the question using the provided context and examples. "
		else:
			if datalist_context is None:
				question_subfix= ""
			else:
				question_subfix= "Answer the question using the provided examples. "
			
		question_subfix+= "Report only the identified class label, without any additional explanation text."
	
	elif prompt_version=="v2":
		description= "You are given an astronomical input image taken from an optical survey. Answer to the question below, strictly following the provided instructions. \n"
		
		if add_task_description:
			context= "\n ## Context: \n"
			context+= "- SMOOTH_ROUND GALAXY: smooth and round galaxy. Should not have signs of spires. \n"
			context+= "- SMOOTH_CIGAR GALAXY: smooth and cigar-shaped galaxy, looks like being seen edge on. This should not have signs of spires of a spiral galaxy. \n"
			context+= "- EDGE_ON_DISK GALAXY: edge-on-disk/spiral galaxy. This disk galaxy should have signs of spires, as seen from an edge-on perspective. \n"
			context+= "- UNBARRED_SPIRAL: unbarred spiral galaxy. Has signs of a disk and/or spires \n"
			context+= "Note that categories SMOOTH_CIGAR and EDGE_ON_DISK classes tend to be very similar to each other. To categorize them, ask yourself the following question: Is this galaxy very smooth, maybe with a small bulge? Then it belongs to class SMOOTH_CIGAR. Does it have irregularities/signs of structure? Then it belongs to class EDGE_ON_DISK."
			description+= context
		
		question_prefix= "\n ## Question: Which of these morphological classes of optical galaxy do you see in the image? \n"
	
		question_subfix= "\n ## Instructions: \n"
		if add_task_description:
			if datalist_context is None:
				question_subfix+= "- Answer the question taking into account the provided context. \n"
			else:
				question_subfix+= "- Answer the question taking into account the provided context and examples. \n"
		else:
			if datalist_context is not None:
				question_subfix+= "- Answer the question taking into account the provided examples. \n"
			
		question_subfix+= "- Answer concisely reporting only the identified source class label taken from these possible choices: SMOOTH_ROUND, SMOOTH_CIGAR, EDGE_ON_DISK, UNBARRED_SPIRAL. DO NOT ADD ANY EXPLANATION TEXT. \n"
		
	else:
		logger.error("Invalid prompt version (%s) specified!" % (prompt_version))
		return
		
	# - Define class options
	class_options= ["SMOOTH_ROUND", "SMOOTH_CIGAR","EDGE_ON_DISK","UNBARRED_SPIRAL"]
	
	label2id= {
		"NONE": 0,
		"SMOOTH_ROUND": 1,
		"SMOOTH_CIGAR": 2,
		"EDGE_ON_DISK": 3,
		"UNBARRED_SPIRAL": 4
	}
	
	task_info= {
		"description": description,
		"question_prefix": question_prefix,
		"question_subfix": question_subfix,
		"classification_mode": "multiclass_singlelabel",
		"label_modifier_fcn": None,
		"label2id": label2id,
		"class_options": class_options
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
	
	
	
