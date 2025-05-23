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
from radio_llava import logger
#logger = logging.getLogger(__name__)

######################
##   LOAD MODEL
######################
def load_llavaov_model_hf(model_name_or_path, device_map="auto", to_float16=False, low_cpu_mem_usage=True, use_attention=False):
	""" Load LLaVA One Vision model """

	# - Load the model in half-precision
	logger.info("Loading model %s ..." % (model_name_or_path))
	if to_float16:
		logger.info("Loading model with float16 torch type ...")
		if use_attention:
			model = LlavaOnevisionForConditionalGeneration.from_pretrained(
				model_name_or_path, 
				torch_dtype=torch.float16, 
				low_cpu_mem_usage=low_cpu_mem_usage,
				#use_flash_attention_2=True,
				attn_implementation="flash_attention_2",
				vision_config={"torch_dtype": torch.float16},
				device_map=device_map
		)
		
		else:
			model = LlavaOnevisionForConditionalGeneration.from_pretrained(
				model_name_or_path, 
				torch_dtype=torch.float16, 
				low_cpu_mem_usage=low_cpu_mem_usage,
				use_flash_attention_2=False,
				device_map=device_map
			)

	else:
		# Attention 2 works with float16 only
		model = LlavaOnevisionForConditionalGeneration.from_pretrained(
			model_name_or_path, 
			low_cpu_mem_usage=low_cpu_mem_usage,
			device_map=device_map
		)
	
	# - To fix warning: Setting `pad_token_id` to `eos_token_id`:None for open-end generation
	model.generation_config.pad_token_id = model.generation_config.eos_token_id
	model.eval()

	# - Load processor
	logger.info("Loading processor for model %s ..." % (model_name_or_path))
	processor = AutoProcessor.from_pretrained(model_name_or_path)
	
	return model, processor	
	

####################################
##   INFERENCE UTILS (HF version)
####################################
def run_llavaov_model_query_hf(
	model,
	processor, 
	image, 
	query,
	do_sample=False,
	temperature=0.2,
	max_new_tokens=4096, 
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
	inputs= processor(
		image, 
		prompt, 
		return_tensors="pt"
	).to(model.device, model.dtype)
	
	#inputs = processor.apply_chat_template(
	#	conversation, 
	#	add_generation_prompt=True, 
	#	tokenize=True, 
	#	return_dict=True, 
	#	return_tensors="pt"
	#).to(model.device, model.dtype)
	
	
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
	
	output = model.generate(
		**inputs, 
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens,
		use_cache=True,
	)
		
	# - Decode response
	output_parsed= processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
	output_parsed_list= output_parsed.split("assistant")
	
	# - Extract predicted label
	response= output_parsed_list[-1].strip("\n").strip()
		
	if verbose:
		print("output")
		print(output)
		
		print("output[0]")
		print(output[0])

		print("output_parsed")
		print(output_parsed)
			
		print("output_parsed (split assistant)")
		print(output_parsed_list)
		
		print("response")
		print(response)
		
	return response


def run_llavaov_model_context_query_hf(
	model,
	processor, 
	image, 
	query,
	images_context,
	conversations_context,
	do_sample=False,
	temperature=0.2,
	max_new_tokens=4096, 
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
		
	inputs = processor(
		images, 
		prompts, 
		padding=True, 
		return_tensors="pt"
	).to(model.device, model.dtype)
		
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
	


def run_llavaov_model_inference_hf(
	datalist, 
	model, processor, 
	task_info, 
	datalist_context=None,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	add_options=False, shuffle_options=False, 
	nmax=-1,
	nmax_context=-1,
	do_sample=False,
	temperature=0.2,
	max_new_tokens=4096,
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
	n_max_retries= 1
	
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
			
		question_retry= "The format of your response does not comply with the requested instructions, please answer again to the following request and strictly follow the given instructions. \n" + question
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
				output= run_llavaov_model_context_query_hf(
					model, processor, 
					image, 
					question,
					images_context,
					conversations_context,
					do_sample=do_sample,
					temperature=temperature if do_sample else None,
					max_new_tokens=max_new_tokens,
					verbose=verbose
				)
			
			else:
				output= run_llavaov_model_query_hf(
					model, processor, 
					image, 
					question,
					do_sample=do_sample,
					temperature=temperature if do_sample else None,
					max_new_tokens=max_new_tokens,
					verbose=verbose
				)
				
			if output is None:
				logger.warn("Failed inference for image %s, skipping ..." % (filename))
				skip_inference= True
				break

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
def run_llavaov_model_rgz_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=False
	)
	

def run_llavaov_model_smorph_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None, 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
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
			
	#question_subfix= "Please report the identified class labels separated by commas, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	question_subfix+= "Report the identified class labels separated by commas, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	

def run_llavaov_model_galaxy_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None, 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run LLaVA inference on radio image dataset (galaxy detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	description= ""
	question_prefix= "Do you see any likely radio galaxy with an extended morphology in the image? "
	question_subfix= "Answer concisely: Yes or No."
	
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	


def run_llavaov_model_artefact_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None, 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run LLaVA inference on radio image dataset (artefact detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	description= ""
	question_prefix= "Do you see any imaging artefact with a ring pattern around bright sources in the image? "
	question_subfix= "Answer concisely: Yes or No."
	
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	

	
def run_llavaov_model_anomaly_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None, 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run LLaVA inference on radio image dataset (anomaly detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	
def run_llavaov_model_mirabest_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None, 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run LLaVA One Vision inference on Mirabest dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	

def run_llavaov_model_gmnist_inference_hf(
	datalist, 
	model, processor, 
	datalist_context=None, 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run LLaVA One Vision inference on Galaxy MNIST dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
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
	return run_llavaov_model_inference_hf(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context,
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)

