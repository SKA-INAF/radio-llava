#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import sys
import json
import argparse
import random
import logging
import shortuuid
import re

from astropy.io import ascii

## MODULE
from radio_llava.utils import *
from radio_llava.inference_llama import *
from radio_llava.inference_internvl import *

## LOGGER
from radio_llava import logger


#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
		
###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input data json filelist') 
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, default=-1, type=int, help='Max number of processed images') 
	
	# - Run options
	parser.add_argument('-nwords_thr','--nwords_thr', dest='nwords_thr', required=False, default=30, type=int, help='Number of words in caption to be used as criterium to sample from short or long description question')
	parser.add_argument('--add_image_description', dest='add_image_description', action='store_true', help='Add image description in the dataset (default=false)')	
	parser.set_defaults(add_image_description=False)
	parser.add_argument('--generate_text_variations', dest='generate_text_variations', action='store_true', help='Generate text variations using LLAMA model (default=false)')	
	parser.set_defaults(generate_text_variations=False)
	parser.add_argument('--generate_qa', dest='generate_qa', action='store_true', help='Add image generated question-answer in the dataset (default=false)')	
	parser.set_defaults(generate_qa=False)
	
	# - Model options
	
	parser.add_argument('-model','--model', dest='model', required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help='LLAMA model used to generate variations') 
	parser.add_argument('-model_type','--model_type', dest='model_type', required=False, default="llama", type=str, help='Model to be used {llama, llama-vision, internvl}') 
	parser.add_argument('-model_name','--model_name', dest='model_name', required=False, type=str, default="", help='InternVL pretrained model name (e.g. InternVL2_5-1B, ...). This is needed for split device_map.') 
	parser.add_argument('-device_map','--device_map', dest='device_map', required=False, default="auto", type=str, help='Device map used when loading model {auto,split}') 
	parser.add_argument('-max_new_tokens','--max_new_tokens', dest='max_new_tokens', required=False, default=1024, type=int, help='The max number of tokens to be generated') 
	parser.add_argument('--do_sample', dest='do_sample', action='store_true', help='Sample LLM responses with temperature parameters (default=false)')	
	parser.set_defaults(do_sample=False)
	parser.add_argument('-top_p','--top_p', dest='top_p', required=False, default=1.0, type=float, help='If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation') 
	parser.add_argument('-top_k','--top_k', dest='top_k', required=False, default=20, type=int, help='The number of highest probability vocabulary tokens to keep for top-k-filtering') 
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, default=0.2, type=float, help='Temperature parameter') 
	parser.add_argument('-penalty','--penalty', dest='penalty', required=False, default=1.2, type=float, help='The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0 and 1.0 penalizes prompt tokens') 
	
	# - Image options
	parser.add_argument('--resize', dest='resize', action='store_true',help='Resize input image (default=false)')	
	parser.set_defaults(resize=False)
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels')
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--contrast', default=0.25, type=float, help='zscale contrast (default=0.25)')
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, default="dump.json", type=str, help='Output data file') 
	
	args = parser.parse_args()	

	return args
	
	
##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	inputfile= args.inputfile		
	outfile= args.outfile
	generate_text_variations= args.generate_text_variations
	model_id= args.model
		
	#===========================
	#==   READ DATA
	#===========================	
	# - Read datalist
	logger.info("Read datalist %s ..." % (inputfile))
	f= open(inputfile, "r")
	datalist= json.load(f)["data"]
	
	#===========================
	#==   LOAD MODEL
	#===========================
	model= None
	tokenizer= None
	processor= None
	if generate_text_variations or args.generate_qa:
		logger.info("Loading model %s ..." % (model_id))
		if args.model_type=="llama":
			model, tokenizer= load_llama_model(model_id, args.device_map)
		elif args.model_type=="llama-vision":
			model, processor= load_llama_vision_model(model_id)
		elif args.model_type=="internvl":
			model, tokenizer= load_internvl_model(model_id, model_name=args.model_name, device_map=args.device_map)
		else:
			logger.error("Invalid/unknown model_type specified (%s)!" % (args.model_type))
			return 1
		
	#===========================
	#==   PROCESS DATA
	#===========================	
	# - Loop over data list and format data
	logger.info("Loop over data list and format data ...")
	outdata= []
	
	description_list = [
		"Can you describe the image?",
		"Describe the image concisely.",
		"Provide a brief description of the given image.",
		"Offer a succinct explanation of the picture presented.",
		"Summarize the visual content of the image.",
		"Give a short and clear explanation of the subsequent image.",
		"Share a concise interpretation of the image provided.",
		"Present a compact description of the images's key features.",
		"Relay a brief, clear account of the picture shown.",
		"Render a clear and concise summary of the image.",
		"Write a terse but informative summary of the image.",
		"Create a compact narrative representing the image presented."
	]
	
	long_description_list = [
		"Can you describe the image in detail?",
		"Describe the following image in detail",
		"Provide a detailed description of the given image",
		"Give an elaborate explanation of the image you see",
		"Share a comprehensive rundown of the presented image",
		"Offer a thorough analysis of the image",
		"Explain the various aspects of the image before you",
		"Clarify the contents of the displayed image with great detail",
		"Characterize the image using a well-detailed description",
		"Break down the elements of the image in a detailed manner",
		"Walk through the important details of the image",
		"Portray the image with a rich, descriptive narrative",
		"Narrate the contents of the image with precision",
		"Analyze the image in a comprehensive and detailed manner",
		"Illustrate the image through a descriptive explanation",
		"Examine the image closely and share its details",
		"Write an exhaustive depiction of the given image"
	]
	
	context= "## Context: You are an AI assistant specialized in radio astronomical topics. You are given an input image from a scientific research paper along with its corresponding text description (Figure caption) provided below: \n"
	
	task= "## Task: Create multiple precise and self-contained question-answer pairs about the input image using the provided image, context and caption text description. For the question-answer generation you must precisely follow the task requirements described below: \n"
	
	task_requirements= "## Task requirements: Below are requirements for generating the questions and answers in the conversation: \n"
	task_requirements+= "- Adopt an astronomical scientific style in both question formulation and question answers. \n"
	task_requirements+= "- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information. \n"
	task_requirements+= "- Do not use phrases like \"mentioned\", \"caption\", \"context\" in the conversation. Instead, refer to the information as being \"in the image\". \n"
	task_requirements+= "- Ensure that questions are diverse and cover a range of visual aspects of the image. \n"
	task_requirements+= "- The conversation should include at least 2-3 turns of questions and answers about the visual aspects of the image. \n"
	task_requirements+= "- Answer responsibly, without inventing words or sentences that deviate or distort the original figure context and description meaning. \n"
	task_requirements+= "- Answers should be clear, specific, and provide comprehensive information based on the image and its provided context/description. \n"
	task_requirements+= "- Ensure that each question-answer pair incorporates all necessary context, allowing them to be fully understood on their own without external references. \n"
	task_requirements+= "- Return generated question-answer pairs using the following json string output format: \n"
	task_requirements+= "[" + "\n"
	task_requirements+= "  " + "{" + "\n"
	task_requirements+= "    " + "\"question\": \"INSERT QUESTION\"," + "\n"
	task_requirements+= "    " + "\"answer\": \"INSERT ANSWER\"" + "\n"
	task_requirements+= "  " + "}," + "\n"
	task_requirements+= "  " + "..." + "\n"
	task_requirements+= "  " + "..." + "\n"
	task_requirements+= "  " + "{" + "\n"
	task_requirements+= "    " + "\"question\": \"INSERT QUESTION\"," + "\n"
	task_requirements+= "    " + "\"answer\": \"INSERT ANSWER\"" + "\n"
	task_requirements+= "  " + "}" + "\n"
	task_requirements+= "]" + "\n"
	task_requirements+= "\n"
	task_requirements+= "DO NOT WRAP THE JSON OUTPUT WITHIN JSON MARKDOWN MARKERS."
	
	for idx, item in enumerate(datalist):
		# - Check stop condition
		if args.nmax!=-1 and idx>=args.nmax:
			logger.info("Stop loop condition reached (%d), as #%d entries were processed..." % (args.nmax, idx))
			break
			
		# - Extract paper info
		filename_paper= item["filepath"]
		figures= item["figures"]
		nfigs= len(figures)
		
		# - Process images in this paper
		logger.info("Processing image %d/%d (paper: %s) ..." % (idx, len(datalist), filename_paper))	
		
		for figitem in figures:
			uuid = shortuuid.uuid()
			image_path= figitem["figure_path"]
			fig_caption= figitem["caption_text"]
			if fig_caption=="":
				logger.warn("Skipping fig %s (paper: %s) with empty caption ..." % (image_path, filename_paper))
				continue
				
			nwords= figitem.get("caption_text_nwords", -1)
				
			# - Initialize outdict
			outdict = dict()
			outdict['id'] = uuid
			outdict['image'] = image_path
	
			# - Fill message conversations
			conversations = []
			
			# ---------------------------------------
			# - Image description
			# .......................................
			if args.add_image_description:
				logger.info("Adding image description for fig %s (paper: %s)" % (image_path, filename_paper))
				
				description_question_sampled= random.choice(description_list)
				if nwords!=-1 and nwords>args.nwords_thr:
					description_question_sampled= random.choice(long_description_list)
			
				q1= {"from": "human", "value": "<image>\n" + description_question_sampled}
			
				description= fig_caption
			
				# - Generate a caption variation?
				if generate_text_variations:
					if args.model_type=="llama-vision":
						description_variant= generate_llama_vision_alternative_text(
							description,
							image_path,
							model, 
							processor,
							temperature=args.temperature,
							max_new_tokens=args.max_new_tokens,
							top_p=args.top_p,
							top_k=args.top_k,
							penalty=args.penalty,
							resize=args.resize, resize_size=args.imgsize,
							zscale=args.zscale, contrast=args.contrast
						)
					elif args.model_type=="internvl":
						description_variant= generate_internvl_alternative_text(
							description,
							image_path, 
							model, 
							tokenizer,
							temperature=args.temperature,
							max_new_tokens=args.max_new_tokens,
							resize_size=args.imgsize,
							zscale=args.zscale, contrast=args.contrast	
						)
					else:
						description_variant= generate_internvl_alternative_text(
							description,
							image_path, 
							model, 
							tokenizer,
							temperature=args.temperature,
							max_new_tokens=args.max_new_tokens,
							resize_size=args.imgsize,
							zscale=args.zscale, contrast=args.contrast	
						)
				
					description= description_variant.strip('\n')
					logger.info("Generated description for fig %s (paper: %s): %s" % (image_path, filename_paper, description))
				
				a1= {"from": "gpt", "value": description}
			
				# - Add messages to collection
				conversations.append(q1)
				conversations.append(a1)
			
			# ---------------------------------------
			# - Multi-turn questions-answers
			# .......................................
			if args.generate_qa:
				query= context + "\n"
				query+= "Figure caption: " + fig_caption + "\n\n"
				query+= task + "\n"
				query+= task_requirements	
			
				response= ""
				if args.model_type=="llama-vision":
					response= run_llama_vision_model_query(
						query,
						image_path,
						model,
						processor,
						do_sample=args.do_sample,
						temperature=args.temperature,
						max_new_tokens=args.max_new_tokens,
						top_p=args.top_p,
						top_k=args.top_k,
						penalty=args.penalty,
						resize=args.resize, resize_size=args.imgsize,
						zscale=args.zscale, contrast=args.contrast,
						verbose=False
					)
				elif args.model_type=="internvl":
					response= run_internvl_model_query(
						model,
						tokenizer,
						image_path, 
						query,
						resize_size=args.imgsize,
						zscale=args.zscale, contrast=args.contrast,
						do_sample=args.do_sample,
						temperature=args.temperature,
						max_new_tokens=args.max_new_tokens,
						verbose=False
					)
				else:
					response= run_internvl_model_query(
						model,
						tokenizer,
						image_path, 
						query,
						resize_size=args.imgsize,
						zscale=args.zscale, contrast=args.contrast,
						do_sample=args.do_sample,
						temperature=args.temperature,
						max_new_tokens=args.max_new_tokens,
						verbose=False
					)
					
				# - Parse model response
				logger.info("Generated multi-turn Q&A for fig %s (paper: %s): %s" % (image_path, filename_paper, response))
				response_str= clean_json_string(response.rstrip())
				
				try:
					responses = json.loads(response_str)
				except Exception as e:
					logger.warning("Failed to convert json string response (%s) for fig %s (paper: %s) to dict list (err=%s), skip image ..." % (response_str, image_path, filename_paper, str(e)))
					continue
			
				# - Extract question-answer pairs
				parsed_questions= []
				parsed_answers= []
				
				if not isinstance(responses,list):
					logger.warning("Converted json string response (%s) for fig %s (paper: %s) is not a dict list, skip image ..." % (response_str, image_path, filename_paper))
					continue
					
				if not responses:
					logger.warning("Converted json string response (%s) for fig %s (paper: %s) is an empty dict list, skip image ..." % (response_str, image_path, filename_paper))
					continue
				
				for index, qaentry in enumerate(responses):
					if not isinstance(qaentry, dict):
						logger.warning("Read item of Q&A list for fig %s (paper: %s) is not a dict, skip Q&A ...")
						continue
					
					if 'question' not in qaentry or 'answer' not in qaentry:
						logger.warning("Read item of Q&A list for fig %s (paper: %s) do not have question/answer keys, skip Q&A ...")
						continue
						
					question= qaentry["question"]
					answer= qaentry["answer"]
					parsed_questions.append(question)
					parsed_answers.append(answer)
				
					if args.add_image_description:
						q_curr= {"from": "human", "value": question}
						a_curr= {"from": "gpt", "value": answer}
					else:
						if index==0:
							q_curr= {"from": "human", "value": "<image>\n" + question}
							a_curr= {"from": "gpt", "value": answer}
						else:
							q_curr= {"from": "human", "value": question}
							a_curr= {"from": "gpt", "value": answer}
						
					# - Add messages to collection
					conversations.append(q_curr)
					conversations.append(a_curr)
					
				logger.info("--> #%d Q&A entries generated and parsed for fig %s (paper: %s) ..." % (len(parsed_questions), image_path, filename_paper))
							
			# - Fill final conversations
			outdict["conversations"]= conversations
	
			# - Append to outdata
			outdata.append(outdict)
	
	# - Convert and write JSON object to file
	logger.info("Convert and write JSON object to file %s ..." % (outfile))
	with open(outfile, "w") as fw: 
		json.dump(outdata, fw, indent=2)	

	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())	
	
