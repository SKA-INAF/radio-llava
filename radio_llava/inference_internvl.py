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
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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
def split_model(model_name):
	""" Compute device map for each model """
	
	device_map = {}
	world_size = torch.cuda.device_count()
	num_layers = {
		'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
		'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80
	}[model_name]

	# Since the first GPU will be used for ViT, treat it as half a GPU.
	num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
	num_layers_per_gpu = [num_layers_per_gpu] * world_size
	num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
	layer_cnt = 0
	for i, num_layer in enumerate(num_layers_per_gpu):
		for j in range(num_layer):
			device_map[f'language_model.model.layers.{layer_cnt}'] = i
			layer_cnt += 1

	device_map['vision_model'] = 0
	device_map['mlp1'] = 0
	device_map['language_model.model.tok_embeddings'] = 0
	device_map['language_model.model.embed_tokens'] = 0
	device_map['language_model.output'] = 0
	device_map['language_model.model.norm'] = 0
	device_map['language_model.lm_head'] = 0
	device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

	return device_map


def load_internvl_model(model_name_or_path, model_name="", device_map="auto"):
	""" Load InternVL model """

	# - Retrieve device map for given model
	if device_map=="split":
		if model_name=="":
			logger.error("model_name is empty, must be set if device_map!=auto!")
			return None
		logger.info("Retrieving device_auto for model %s ..." % (model_name))
		device_map = split_model(model_name)
	
	# - Load the model
	logger.info("Loading model %s ..." % (model_name_or_path))
	model = AutoModel.from_pretrained(
		model_name_or_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map
   ).eval()
   #.cuda()
   
	# - Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)

	return model, tokenizer

####################################
##   INFERENCE UTILS
####################################
def run_internvl_model_query(
	model,
	tokenizer,
	image_path, 
	query,
	resize_size=448,
	zscale=False, contrast=0.25,
	do_sample=False,
	temperature=0.2,
	verbose=False
):
	""" Run InternVL model inference """
	
	# - Load image
	pixel_values= load_img_as_internvl(
		image_path, 
		resize=False, resize_size=448, 
		apply_zscale=zscale, contrast=contrast, 
		set_nans_to_min=False, 
		verbose=False
	).to(torch.bfloat16).cuda()
	
	# - Set generation config
	num_beams= 1
	top_p= None
	max_new_tokens= 1024
	
	generation_config = dict(
		max_new_tokens=max_new_tokens, 
		do_sample=do_sample,
		temperature=temperature if do_sample else None
	)

	# - Set question
	question= '<image>\n' + query
	
	# - Run inference
	output = model.chat(
		tokenizer, 
		pixel_values, 
		question, 
		generation_config
	)

	if verbose:
		print("output")
		print(output)
		
	# - Format response
	response= output.strip("\n").strip()
	
	return response
	
def run_internvl_model_inference(
	datalist, 
	model,
	tokenizer,
	task_info,
	resize_size=448, 
	zscale=False, contrast=0.25,
	add_options=False, shuffle_options=False, nmax=-1, 
	verbose=False
):
	""" Run InternVL inference on radio image dataset """

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

	#===========================
	#==   RUN INFERENCE
	#===========================
	# - Loop over images in dataset
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
		
		# - Create question
		if add_options:
			option_choices= class_options.copy()
			if shuffle_options:
				random.shuffle(option_choices)
		
			question_labels= ' \n '.join(option_choices)		
			question= description + ' \n' + question_prefix + ' \n ' + question_labels + question_subfix
		else:
			question= description + ' \n' + question_prefix + ' \n ' + question_subfix
			
		# - Query model
		output= run_internvl_model_query(
			model=model,
			tokenizer=tokenizer,
			image_path=filename,
			query=question,
			resize_size=resize_size, 
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
def run_internvl_model_rgz_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1,
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on RGZ dataset """

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
		question_subfix= "Answer the question using the provided context. "
	else:
		question_subfix= ""
			
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		verbose=verbose
	)	
	
def run_internvl_model_smorph_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on radio image dataset """
	
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
		question_subfix= "Answer the question using the provided context. "
	else:
		question_subfix= ""
			
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		verbose=verbose
	)		
	
	
def run_internvl_model_galaxy_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on radio image dataset (galaxy detection) """
	
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		verbose=verbose
	)
	
	
def run_internvl_model_artefact_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on radio image dataset (artefact detection) """
	
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		verbose=verbose
	)	
	
def run_internvl_model_anomaly_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on radio image dataset (anomaly detection) """
	
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
		question_subfix= "Answer the question using the provided context. "
	else:
		question_subfix= ""
		
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=False, shuffle_options=False, 
		nmax=nmax, 
		verbose=verbose
	)
	
def run_internvl_model_mirabest_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1,
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on Mirabest dataset """

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
		question_subfix= "Answer the question using the provided context. "
	else:
		question_subfix= ""
		
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		verbose=verbose
	)
	
def run_internvl_model_gmnist_inference(
	datalist, 
	model, tokenizer, 
	resize_size=448, 
	zscale=False, contrast=0.25, 
	shuffle_options=False, 
	nmax=-1,
	add_task_description=False,
	verbose=False
):
	""" Run InternVL inference on Galaxy MNIST dataset """

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
		question_subfix= "Answer the question using the provided context. "
	else:
		question_subfix= ""
		
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
	return run_internvl_model_inference(
		datalist, 
		model, tokenizer, 
		task_info, 
		resize_size=resize_size, 
		zscale=zscale, contrast=contrast, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		verbose=verbose
	)
