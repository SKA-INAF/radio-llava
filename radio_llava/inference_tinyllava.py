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
from radio_llava.inference_utils import *

## LOGGER
logger = logging.getLogger(__name__)


######################
##   LOAD MODEL
######################
def load_tinyllava_model_lora(model_name_or_path, device="cuda", set_float16=False):
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
	if set_float16:
		logger.info("Loading model to device with float16 type ...")
		model.to(device, torch.float16)
	else:
		model.to(device)
	
	# - Merge LORA weights into model
	logger.debug("Loading LoRA weights...")
	model = PeftModel.from_pretrained(model, model_name_or_path)
	logger.debug("Merging LoRA weights...")
	model = model.merge_and_unload()
	logger.debug("Model is loaded...")
	
	return model
		

def load_tinyllava_model_standard(model_name_or_path, device="cuda", set_float16=False):
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
	if set_float16:
		logger.info("Loading model to device with float16 type ...")
		model.to(device, torch.float16)
	else:
		model.to(device)
	
	return model


def load_tinyllava_model(
	model_name_or_path,
	load_lora_model=False,
	device="cuda",
	set_float16=False
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
	disable_torch_init()
	
	if load_lora_model:
		model= load_tinyllava_model_lora(model_name_or_path, device, set_float16)
	else:
		try:
			if set_float16:
				model = TinyLlavaForConditionalGeneration.from_pretrained(
					model_name_or_path,
					low_cpu_mem_usage=True,
					torch_dtype=torch.float16,
					device_map="auto"
				)
			else:
				model = TinyLlavaForConditionalGeneration.from_pretrained(
					model_name_or_path,
					low_cpu_mem_usage=True,
					device_map="auto"
				)
		except Exception as e:
			logger.warn("Failed to load pre-trained model (err=%s), trying with another method ..." % (str(e)))
			model= load_tinyllava_model_standard(model_name_or_path, device, set_float16)
 	
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
	#input_ids= input_ids.unsqueeze(0).cuda()
	input_ids= input_ids.unsqueeze(0).to(model.device)

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
		logger.warn("Failed to read image %s, returning None ..." % (image_path))
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
	#image_tensor= image_tensor.unsqueeze(0).half().cuda()
	image_tensor= image_tensor.unsqueeze(0).to(model.device, torch.float16)
	
	if verbose:
		print("image_tensor.shape")
		print(image_tensor.shape)

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
		temperature=temperature if do_sample else None,
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
	)
	
	if verbose:
		print("outputs")
		print(outputs)
		
	outputs= outputs[0].strip()
	
	if verbose:
		print("outputs (after strip)")
		print(outputs)
	
	if outputs.endswith(stop_str):
		outputs = outputs[: -len(stop_str)]
		if verbose:
			print("outputs (endswith) (stop_str=%s)" % (stop_str))
			print(outputs)
		
	outputs = outputs.strip()
	
	if verbose:
		print("outputs (final)")
		print(outputs)
	  
	return outputs


def run_tinyllava_model_inference(
	datalist, 
	model, 
	task_info,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi',
	add_options=False, shuffle_options=False, nmax=-1, 
	verbose=False
):
	""" Run TinyLLaVA inference on radio image dataset """

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
def run_tinyllava_model_rgz_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	shuffle_options=False, nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run inference on RGZ dataset """

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
	
	#question_subfix+= "Answer the question reporting only the identified class label, without any additional explanation text."
	question_subfix+= "Report only the identified class label, without any additional explanation text."
	
	#if add_task_description:
	#	description= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	#else:
	#	description= ""
		
	#question_prefix= "Which of these morphological classes of radio sources do you see in the image? "
	#question_subfix= "Please report only the identified class label, without any additional explanation text. Report just NONE if you cannot recognize any of the above classes in the image."
	
	
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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode, 
		add_options=True, shuffle_options=shuffle_options, nmax=nmax, 
		verbose=verbose
	)
	
	
def run_tinyllava_model_smorph_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	shuffle_options=False, nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run TinyLLaVA inference on radio image dataset """
	
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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode, 
		add_options=True, shuffle_options=shuffle_options, nmax=nmax, 
		verbose=verbose
	)	
	

def run_tinyllava_model_galaxy_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	nmax=-1, 
	verbose=False
):
	""" Run TinyLLaVA inference on radio image dataset (galaxy detection) """
	
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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode,
		add_options=False, shuffle_options=False,
		nmax=nmax, 
		verbose=verbose
	)	
	

def run_tinyllava_model_artefact_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	nmax=-1, 
	verbose=False
):
	""" Run TinyLLaVA inference on radio image dataset (artefact detection) """
	
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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode, 
		add_options=False, shuffle_options=False,
		nmax=nmax, 
		verbose=verbose
	)	
	
def run_tinyllava_model_anomaly_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	shuffle_options=False,
	nmax=-1,
	add_task_description=False,
	verbose=False
):
	""" Run TinyLLaVA inference on radio image dataset (anomaly detection) """
	
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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode, 
		add_options=True, shuffle_options=shuffle_options,
		nmax=nmax, 
		verbose=verbose
	)	
	




def run_tinyllava_model_mirabest_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	shuffle_options=False, nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run inference on Mirabest dataset """

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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode, 
		add_options=True, shuffle_options=shuffle_options, 
		nmax=nmax, 
		verbose=verbose
	)

def run_tinyllava_model_gmnist_inference(
	datalist, 
	model,
	reset_imgnorm=False,
	resize=False, resize_size=384, 
	zscale=False, contrast=0.25,
	conv_mode='phi', 
	shuffle_options=False, nmax=-1, 
	add_task_description=False,
	verbose=False
):
	""" Run TinyLLaVA inference on Galaxy MNIST dataset """

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
	return run_tinyllava_model_inference(
		datalist, 
		model, 
		task_info, 
		reset_imgnorm=reset_imgnorm, 
		resize=resize, resize_size=resize_size, 
		zscale=zscale, contrast=contrast,
		conv_mode=conv_mode, 
		add_options=True, shuffle_options=shuffle_options, nmax=nmax, 
		verbose=verbose
	)
	
