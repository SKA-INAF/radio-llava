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

## QWENVL MODULES
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

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
def load_qwen2vl_model(model_name_or_path, device_map="auto", imgsize_min=64, imgsize_max=256):
	""" Load Qwen2VL model """

	# - Load the model
	logger.info("Loading model %s ..." % (model_name_or_path))
	model = Qwen2VLForConditionalGeneration.from_pretrained(
		model_name_or_path,
		torch_dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
		device_map=device_map
	)
	
	# - Load processor
	# The default range for the number of visual tokens per image in the model is 4-16384.
	# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
	logger.info("Loading processor from model %s ..." % (model_name_or_path))
	min_pixels = imgsize_min*28*28
	max_pixels = imgsize_max*28*28
	processor = AutoProcessor.from_pretrained(
		model_name_or_path,
		min_pixels=min_pixels,
		max_pixels=max_pixels
	)
	
	return model, processor
	
####################################
##   INFERENCE UTILS
####################################
def run_qwen2vl_model_query(
	model,
	processor,
	image, 
	query,
	do_sample=False,
	temperature=0.2,
	verbose=False
):
	""" Run Qwen2VL model inference """  
	
	# - Define message prompt
	messages = [
  	{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": image,
				},
				{
					"type": "text",
					"text": query
      	}
    	]
  	}
	]
	
	# - Set text template
	text = processor.apply_chat_template(
		messages, 
		tokenize=False, 
		add_generation_prompt=True
	)
	
	# - Process image
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
	  text=[text],
	  images=image_inputs,
	  videos=video_inputs,
	  padding=True,
	  return_tensors="pt",
	)
	inputs= inputs.to(model.device)

	# - Inference: Generation of the output
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 128
	
	output = model.generate(
		**inputs, 
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens
	)
	output_trimmed = [
  	out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
	]

	# - Decode outputs
	logger.debug("Decode outputs ...")
	output_parsed = processor.batch_decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	
	if verbose:
		print("output")
		print(output)
		
		print("output_trimmed")
		print(output_trimmed)

		print("output_parsed")
		print(output_parsed)
			
	# - Extract predicted label
	response= output_parsed[0].strip("\n").strip()
	
	return response

def run_qwen2vl_model_context_query(
	model,
	processor,
	image, 
	query,
	images_context,
	conversations_context,
	do_sample=False,
	temperature=0.2,
	verbose=False
):
	""" Run Qwen2VL model inference """  

	# - Check context info
	if not images_context or not conversations_context:
		logger.error("Empty list given for either context images, queries or responses!")
		return None
		
	# - Create messages for batch inference
	messages= []
	for i in range(len(conversations_context)):
		image_context= images_context[i]
		item= conversations_context[i]
		query_context= item['question']
		response_context= item['response']
		
		message= [
			{
				"role": "user", 
				"content": [
					{"type": "image", "image": image_context},
					{"type": "text", "text": query_context}
				]
			},
			{
				"role": "system", 
				"content": response_context
			}
		]
		messages.append(message)
		
	message= [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": image},
				{"type": "text", "text": query},
			]
		}
	]	
	messages.append(message)
			
	# - Preparation for batch inference
	texts = [
		processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
		for msg in messages
	]
	
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=texts,
		images=image_inputs,
		videos=video_inputs,
		padding=True,
		return_tensors="pt",
	)
	inputs = inputs.to(model.device)

	# - Inference: Generation of the output
	logger.debug("Generate model response ...")
	num_beams= 1
	top_p= None
	max_new_tokens= 128
	
	output = model.generate(
		**inputs, 
		do_sample=do_sample,
		temperature=temperature if do_sample else None,
		top_p=top_p,
		num_beams=num_beams,
		max_new_tokens=max_new_tokens
	)
	output_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
	]

	# - Decode outputs
	logger.debug("Decode outputs ...")
	output_parsed= processor.batch_decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	
	if verbose:
		print("output")
		print(output)
		
		print("output_trimmed")
		print(output_trimmed)

		print("output_parsed")
		print(output_parsed)
			
	# - Extract predicted label
	response= output_parsed[0].strip("\n").strip()
	
	return response
	
	
	
def run_qwen2vl_model_inference(
	datalist, 
	model, processor, 
	task_info, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	add_options=False, shuffle_options=False,
	nmax=-1,
	nmax_context=-1,
	verbose=False
):
	""" Run Qwen2VL inference on radio image dataset """

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
				
			print("question: ", question_curr)
						
			if conversations_context:
				output= run_qwen2vl_model_context_query(
					model, processor, 
					image, 
					question_curr,
					images_context,
					conversations_context,
					do_sample=False,
					temperature=None,
					verbose=verbose
				)
			else:
				output= run_qwen2vl_model_query(
					model, processor, 
					image, 
					question_curr,
					do_sample=False,
					temperature=None,
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
			print("output:", output)
			print("res: ", res)
			print("type(res)", type(res))
			
			if res is None:
				if n_retries>=n_max_retries:
					logger.warn("Unexpected label prediction obtained for image %s, giving up and skipping image ..." % (filename))
					skip_inference= True
					break
				else:
					n_retries+= 1
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
def run_qwen2vl_model_rgz_inference(
	datalist, 
	model, processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1,
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run Qwen2VL inference on RGZ dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	context= "### Context: Consider these morphological classes of radio astronomical sources: \n 1C-1P: single-island sources having only one flux intensity peak; \n 1C-2C: single-island sources having two flux intensity peaks; \n 1C-3P: single-island sources having three flux intensity peaks; \n 2C-2P: sources consisting of two separated islands, each hosting a single flux intensity peak; \n 2C-3P: sources consisting of two separated islands, one containing a single peak of flux intensity and the other exhibiting two distinct intensity peaks; 3C-3P: sources consisting of three separated islands, each hosting a single flux intensity peak. \n An island is a group of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
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
			
	question_subfix+= "Answer the question reporting only the identified class label, without any additional explanation text."
	
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
	return run_qwen2vl_model_inference(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)


def run_qwen2vl_model_smorph_inference(
	datalist, 
	model, processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run Qwen2VL inference on radio image dataset """
	
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
	return run_qwen2vl_model_inference(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	

def run_qwen2vl_model_galaxy_inference(
	datalist, 
	model, processor,
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run Qwen2VL inference on radio image dataset (galaxy detection) """
	
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
	return run_qwen2vl_model_inference(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	
def run_qwen2vl_model_artefact_inference(
	datalist, 
	model, processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run Qwen2VL inference on radio image dataset (artefact detection) """
	
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
	return run_qwen2vl_model_inference(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	

def run_qwen2vl_model_anomaly_inference(
	datalist, 
	model, processor, 
	datalist_context=None, 
	device="cuda:0", 
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	nmax_context=-1,
	add_task_description=False,
	verbose=False
):
	""" Run Qwen2VL inference on radio image dataset (anomaly detection) """
	
	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	if add_task_description:
		description= "Consider this radio image peculiarity classes, defined as follows: \n ORDINARY: image containing only point-like or slightly-resolved compact radio sources superimposed over the sky background or imaging artefact patterns; \n COMPLEX: image containing one or more radio sources with extended or diffuse morphology; \n PECULIAR: image containing one or more radio sources with anomalous or peculiar extended morphology, often having diffuse edges, complex irregular shapes, covering a large portion of the image.\n"
	else:
		description= ""
	question_prefix= "Can you identify which peculiarity class the presented image belongs to? "
	question_subfix= "Please report only the identified class label, without any additional explanation text."
	
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
	return run_qwen2vl_model_inference(
		datalist, 
		model, processor, 
		task_info, 
		datalist_context=datalist_context, 
		device=device, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		nmax_context=nmax_context,
		verbose=verbose
	)	
	


