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
	

def run_llavaov_model_rgz_inference(datalist, model, processor, datalist_context=None, device="cuda:0", resize=False, resize_size=384, zscale=False, contrast=0.25, shuffle_label_options=False, nmax=-1, verbose=False):
	""" Run LLaVA One Vision inference on RGZ dataset """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	description= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	
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
	
	nclasses= len(label2id)
	class_names= list(label2id.keys())
	labels= list(label2id.values())
	
	#====================================
	#==   CREATE CONTEXT CONVERSATIONS
	#====================================
	# - Create context conversation (if datalist_context if given)
	conversations_context= []
	images_context= []
	
	if datalist_context is not None:
		for idx, item in enumerate(datalist_context):
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
			option_choices= class_names.copy()
			question_labels= ' \n '.join(option_choices)
			if idx==0:
				question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
			else:
				question= question_prefix + ' \n ' + question_labels + question_subfix
		
			# - Set assistant response to true label
			response= label
			
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
	# - Loop over images in dataset
	ninferences_unexpected= 0
	ninferences_failed= 0
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
		option_choices= class_names.copy()
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

		# - Extract predicted label
		label_pred= output.strip("\n").strip().upper()

		# - Check if label is correct
		if label_pred not in label2id:
			logger.warn("Unexpected label (%s) returned, skip this image ..." % (label_pred))
			ninferences_unexpected+= 1
			continue
	
		# - Extract class ids
		classid= label2id[label]
		classid_pred= label2id[label_pred]
		classids.append(classid)
		classids_pred.append(classid_pred)	
		logger.info("image %s: GT(id=%d, label=%s), PRED(id=%d, label=%s)" % (sname, classid, label, classid_pred, label_pred))

	logger.info("#%d failed inferences" % (ninferences_failed))
	logger.info("#%d unexpected inferences" % (ninferences_unexpected))


	#===========================
	#==   COMPUTE METRICS
	#===========================
	# - Compute and print metrics
	y_pred= np.array(classids_pred)
	y_true= np.array(classids)	
	#metrics= multiclass_singlelabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=class_names)
	metrics= multiclass_singlelabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=labels)
	print_metrics(metrics)
		
	return 0

















def run_rgz_data_inference(datalist, model, processor, datalist_context=None, device="cuda:0", resize=False, resize_size=384, zscale=False, contrast=0.25, shuffle_label_options=False, nmax=-1, verbose=False):
	""" Convert RGZ datalist to conversational data """

	#===========================
	#==   INIT TASK
	#===========================
	# - Define message
	description= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	
	question_prefix= "Which of these morphological classes of radio sources do you see in the image? "
	question_subfix= "Please report only the identified class label, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	
	#labels= ["1C-1P","1C-2P","1C-3P","2C-2P","2C-3P","3C-3P"]
	label2id= {
		"1C-1P": 0,
		"1C-2P": 1,
		"1C-3P": 2,
		"2C-2P": 3,
		"2C-3P": 4,
		"3C-3P": 5,
	}
	
	#nclasses= len(labels)
	nclasses= len(label2id)
	class_names= list(label2id.keys())
	labels= list(label2id.values())
	
	#==============================
	#==   CREATE CONTEXT MESSAGE
	#==============================
	# - Create context conversation (if datalist_context if given)
	conversations_context= []
	#prompts_context= []
	images_context= []
	
	if datalist_context is not None:
		for idx, item in enumerate(datalist_context):
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
			option_choices= class_names.copy()
			question_labels= ' \n '.join(option_choices)
			if idx==0:
				question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
			else:
				question= question_prefix + ' \n ' + question_labels + question_subfix
		
			# - Set assistant response to true label
			response= label
			
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
			
		#if verbose:
		#	print("conversations_context")
		#	print(json.dumps(conversations_context, indent=2))
			
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
		#class_id= item["id"]
		label= item["label"]
		
		# - Read image into PIL
		image= load_img_as_pil_rgb(
			filename,
			resize=resize, resize_size=resize_size, 
			apply_zscale=zscale, contrast=contrast,
			verbose=verbose
		)
		if image is None:
			logger.warn("Read image %s is None, skipping inference for this ..." % (filename))
			continue
		
		# - Create question
		option_choices= class_names.copy()
		if shuffle_label_options:
			random.shuffle(option_choices)
		
		question_labels= ' \n '.join(option_choices)
		if conversations_context:
			question= question_prefix + ' \n ' + question_labels + question_subfix
		else:
			question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
		
		# - Create conversation (eventually adding also context conversations)
		conversation= [
			{
				"role": "user",
				"content": [
					{"type": "image"},
					{"type": "text", "text": question},
				],
    	},
		]
		conversations= conversations_context.copy()
		conversations.extend(conversation)
		
		# - Create prompt 
		#prompt= processor.apply_chat_template(conversation, add_generation_prompt=True)
		prompts= processor.apply_chat_template(conversations, add_generation_prompt=True)
		
		# - Create model inputs (eventually combining context and inference prompts)
		images= images_context.copy()
		images.append(image)
		
		#inputs = processor(image, prompt, return_tensors="pt").to(model.device, torch.float16)
		inputs = processor(images, prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
		
		if verbose:
			print("conversations")
			print(json.dumps(conversations, indent=2))
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
		label_pred= output_parsed_list[-1].strip("\n").strip()

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
	#metrics= multiclass_singlelabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=class_names)
	metrics= multiclass_singlelabel_metrics(y_true=y_true, y_pred=y_pred, target_names=class_names, labels=labels)
	print_metrics(metrics)
		
	return 0
	

