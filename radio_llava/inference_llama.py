#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import os
import sys
import json
import argparse
import random
import logging

# - PyTorch
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TextIteratorStreamer, BitsAndBytesConfig
from transformers import MllamaForConditionalGeneration, AutoProcessor

## MODULE
from radio_llava.utils import *

## LOGGER
logger = logging.getLogger(__name__)

######################
##   LOAD MODEL
######################
def load_llama_model(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto"):
	""" Load LLAMA model """

	# - Create model
	quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
	)

	logger.info("Loading tokenizer ...")
	tokenizer = AutoTokenizer.from_pretrained(
		pretrained_model_name_or_path=model_id
	)
	
	logger.info("Loading model ...")
	model = AutoModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path=model_id,
		device_map=device_map,
		torch_dtype=torch.bfloat16,
		quantization_config=quantization_config,
		low_cpu_mem_usage=True,
	)
	
	return model, tokenizer
	

def load_llama_vision_model(model_id="alpindale/Llama-3.2-11B-Vision-Instruct", device_map="auto"):
	""" Load LLAMA vision model """

	# - Load model
	logger.info("Loading model %s ..." % (model_id))
	model = MllamaForConditionalGeneration.from_pretrained(
		model_id,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)
	
	# - Load processor
	logger.info("Loading processor %s ..." % (model_id))
	processor = AutoProcessor.from_pretrained(model_id)
	
	return model, processor

######################
##   QUERY MODEL
######################
def run_llama_model_query(
	prompt, 
	model, 
	tokenizer, 
	do_sample=False,
	temperature=0.2,
	max_new_tokens=1024,
	top_p=1.0,
	top_k=20,
	penalty=1.2
):
	""" Generate response to input text using LLAMA model """
	
	# - Create message conversation
	conversation= [
  	{"role": "user", "content": prompt}
	]
	
	# - Create inputs
	input_ids = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
	tokenizer.pad_token = tokenizer.eos_token
	inputs = tokenizer(input_ids, return_tensors='pt', padding=True, truncation=True).to(model.device)
	
	# - Generate text output
	terminators = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids("<|eot_id|>")
	]
	
	output= model.generate(
		**inputs,
		max_new_tokens = max_new_tokens,
		do_sample = do_sample,
		temperature = temperature if do_sample else None,
		top_p = top_p,
		top_k = top_k,
		repetition_penalty=penalty,
		eos_token_id=terminators,
		pad_token_id=tokenizer.eos_token_id,
		num_return_sequences=1
	)

	# - Decode text output
	response= tokenizer.decode(output[0], skip_special_tokens=True)
	output_text= response.split("assistant")[1]
	
	return output_text
	
	
def generate_llama_alternative_text(
	input_text, 
	model, 
	tokenizer,
	temperature=0.2,
	max_new_tokens=1024,
	top_p=1.0,
	top_k=20,
	penalty=1.2
):
	""" Generate text variation """

	# - Create prompt
	prompt_start= "Generate a variation of the following sentence, keeping the same content, using words that match the original sentence's meaning, and using an astronomical scientific style, without adding additional details: \n" 
	prompt_end= "\n Please report only the best alternative sentence without any prefix, preamble or explanation."
	prompt= prompt_start + input_text + prompt_end

	# - Generate text
	return run_llama_model_query(
		prompt, 
		model, 
		tokenizer, 
		do_sample=True,
		temperature=temperature,
		max_new_tokens=max_new_tokens,
		top_p=top_p,
		top_k=top_k,
		penalty=penalty
	)
	
	
def run_llama_vision_model_query(
	query,
	image_path,
	model,
	processor,
	do_sample=False,
	temperature=0.2,
	max_new_tokens=1024,
	top_p=1.0,
	top_k=20,
	penalty=1.2,
	resize=False, resize_size=384,
	zscale=False, contrast=0.25,
	verbose=False
):
	""" Query LLAMA vision model """
	
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
	
	# - Create message
	messages = [
		{"role": "user", "content": 
			[
				{"type": "image"},
				{"type": "text", "text": query}
    	]
    }
	]

	input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
	inputs = processor(image, input_text, return_tensors="pt").to(model.device)

	# - Generate model response
	logger.debug("Generate model response ...")
	
	#terminators = [
	#	tokenizer.eos_token_id,
	#	tokenizer.convert_tokens_to_ids("<|eot_id|>")
	#]
	
	output = model.generate(
		**inputs, 
		max_new_tokens = max_new_tokens,
		do_sample = do_sample,
		temperature = temperature if do_sample else None,
		#top_p = top_p,
		#top_k = top_k,
		#repetition_penalty=penalty,
	)
	output_decoded= processor.decode(output[0], skip_special_tokens=True)
	
	return output_decoded
	
def generate_llama_vision_alternative_text(
	input_text,
	image_path, 
	model, 
	processor,
	temperature=0.2,
	max_new_tokens=1024,
	top_p=1.0,
	top_k=20,
	penalty=1.2,
	resize=False, resize_size=384,
	zscale=False, contrast=0.25,
	verbose=False
):
	""" Generate text variation """

	# - Create prompt
	prompt_start= "Generate a text variation of the following description of the input image, keeping the same content, using words that match the original sentence's meaning, and using an astronomical scientific style, without adding additional details: \n" 
	prompt_end= "\n Please report only the best alternative sentence without any prefix, preamble or explanation."
	prompt= prompt_start + input_text + prompt_end

	# - Generate text
	return run_llama_vision_model_query(
		prompt,
		image_path, 
		model, 
		processor, 
		do_sample=True,
		temperature=temperature,
		max_new_tokens=max_new_tokens,
		top_p=top_p,
		top_k=top_k,
		penalty=penalty,
		resize=resize, resize_size=resize_size,
		zscale=zscale, contrast=contrast,
		verbose=verbose
	)
	
