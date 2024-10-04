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
	
