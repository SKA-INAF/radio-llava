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
	parser.add_argument('--generate_text_variations', dest='generate_text_variations', action='store_true', help='Generate text variations using LLAMA model (default=false)')	
	parser.set_defaults(generate_text_variations=False)
	parser.add_argument('--generate_qa', dest='generate_qa', action='store_true', help='Add image generated question-answer in the dataset (default=false)')	
	parser.set_defaults(generate_qa=False)
	parser.add_argument('--add_image_description', dest='add_image_description', action='store_true', help='Add image description in the dataset (default=false)')	
	parser.set_defaults(add_image_description=False)
	
	# - Model options
	parser.add_argument('-model','--model', dest='model', required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help='LLAMA model used to generate variations') 
	parser.add_argument('-model_type','--model_type', dest='model_type', required=False, default="llama", type=str, help='Model to be used {llama, llama-vision}') 
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
	#==   LOAD LLAMA MODEL
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
	
	classification_msg_header= "Consider these morphological classes of radio astronomical sources, defined as follows: \n 1C-1P: single-island radio sources having only one flux intensity peak; \n 1C-2C: single-component radio sources having two flux intensity peaks; \n 1C-3P: single-island radio sources having three flux intensity peaks; \n 2C-2P: radio sources formed by two disjoint islands, each hosting a single flux intensity peak; \n 2C-3P: radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks; 3C-3P: radio sources formed by three disjoint islands, each hosting a single flux intensity peak. An island is a group or blob of 4-connected pixels in an image under analysis with intensity above a detection threshold with respect to the sky background level. "
	
	
	context= "## Context: You are an AI assistant specialized in radio astronomical topics. You are given an input image from a scientific research paper along with its corresponding text description (Figure caption) provided below: \n"
	
	glossary= "## Glossary:\n"
	glossary+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level. The terms 'island' and 'component' are sometimes used interchangeably, but an 'island' is defined purely by pixel connectivity above a noise threshold, and may contain one or multiple source components. Example: A radio galaxy with extended lobes may appear as one large source island, encompassing multiple structures (core, jets, lobes).\n"
	glossary+= "- SOURCE COMPONENT: an individual emission structure within a radio astronomical image that is associated with a physically coherent entity or a substructure of a larger source. In source extraction, a single astronomical source may be represented by one or multiple components, depending on its morphology and how the extraction algorithm segments emission regions. These components may correspond to compact sources, such as individual galaxies or quasars, or extended sources, such as lobes of radio galaxies or supernova remnants. Example: A radio galaxy with a central core and two extended lobes may be decomposed into three source components—one for the core and one for each lobe. A single source island may contain multiple source components (e.g., a double-lobed radio galaxy).\n"
	glossary+= "- SOURCE INTENSITY PEAK: the location of the maximum brightness (or flux density) within a detected source component in a radio astronomical image. It represents the highest observed intensity value in the 2D brightness distribution of the source, typically measured in Jy/beam (Jansky per beam). The peak intensity is crucial for identifying the position of the source, as well as for distinguishing between compact and extended emission structures. Example: For a compact radio source, such as a quasar, the intensity peak often coincides with the centroid of the source. In contrast, for an extended radio galaxy, the peak intensity may be located in the core or along the brightest part of the radio lobes. \n"
	glossary+= "- 1C-1P SOURCE: morphological class of single-island radio sources having only one flux intensity peak. \n"
	glossary+= "- 1C-2P SOURCE: morphological class of single-island radio sources having two flux intensity peaks. \n"
	glossary+= "- 1C-3P SOURCE: morphological class of single-island radio sources having three flux intensity peaks. \n"
	glossary+= "- 2C-2P SOURCE: morphological class of radio sources formed by two disjoint islands, each hosting a single flux intensity peak. \n"
	glossary+= "- 2C-3P SOURCE: morphological class of radio sources formed by two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks. \n"
	glossary+= "- 3C-3P SOURCE: morphological class of radio sources formed by three disjoint islands, each hosting a single flux intensity peak. \n"
	glossary+= "\n"
	
	task= "## Task: Create multiple precise and self-contained question-answer pairs about the input image using the provided image, context and caption text description. For the question-answer generation you must precisely follow the task requirements described below: \n"
	
	task_requirements= "## Task requirements: Below are requirements for generating the questions and answers in the conversation: \n"
	task_requirements+= "- Adopt an astronomical scientific style in both question formulation and question answers, following definitions and concepts given in the Glossary. \n"
	task_requirements+= "- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information. \n"
	task_requirements+= "- Do not use phrases like \"mentioned\", \"caption\", \"context\" in the conversation. Instead, refer to the information as being \"in the image\". \n"
	task_requirements+= "- Ensure that questions are diverse and cover a range of visual aspects of the image. \n"
	task_requirements+= "- The conversation should include at least 1 or 2 turns of questions and answers about the visual aspects of the image that fully cover all information reported in the provided caption. \n"
	task_requirements+= "- Answer responsibly, without inventing words or sentences that deviate or distort the original figure context and description meaning. \n"
	task_requirements+= "- Answers should be clear, specific, and provide comprehensive information based on the image and its provided context/description. \n"
	task_requirements+= "- Ensure that each question-answer pair incorporates all necessary context, allowing them to be fully understood on their own without external references. \n"
	task_requirements+= "- Include at least one question to determine what morphological kind of radio source is present in the image. \n"
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
			
		uuid = shortuuid.uuid()
		filename= item["filepaths"][0]
		class_id= item["id"]
		label= item["label"]
		
		print("Processing image %d/%d (%s) ..." % (idx, len(datalist), filename))	
	
		# - Initialize outdict
		outdict = dict()
		outdict['id'] = uuid
		outdict['image'] = filename
	
		# - Fill message conversations
		conversations = []
			
		# ---------------------------------------
		# - Image description
		# .......................................
		q1= {"from": "human", "value": "<image>\n" + random.choice(description_list)}
		
		description= "The image is a radio astronomical image cutout extracted from a larger radio-continuum Stokes-I map produced by an interferometer telescope. "
		if label=="1C-1P":
			description+= "The image is centred and zoomed on a single-island radio source having only one flux intensity peak. "
		elif label=="1C-2P":
			description+= "The image is centred and zoomed on a single-island radio source having two flux intensity peaks. "
		elif label=="1C-3P":
			description+= "The image is centred and zoomed on a single-island radio source having three flux intensity peaks. "
		elif label=="2C-2P":
			description+= "The image is centred and zoomed on a radio source consisting of two disjoint islands, each hosting a single flux intensity peak. "			
		elif label=="2C-3P":
			description+= "The image is centred and zoomed on a radio source consisting of two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks. "
		elif label=="3C-3P":
			description+= "The image is centred and zoomed on a radio source consisting of three disjoint islands, each hosting a single flux intensity peak. "
		
		description+= "An island is a group of 4-connected pixels with flux above a detection threshold with respect to the sky background level. "
		
		description_final= description
		print("description: ", description_final)
		if generate_text_variations:
			if args.model_type=="llama":
				description_final= generate_llama_alternative_text(
					description,
					model, 
					tokenizer,
					temperature=args.temperature,
					max_new_tokens=args.max_new_tokens,
					top_p=args.top_p,
					top_k=args.top_k,
					penalty=args.penalty
				)
			elif args.model_type=="llama-vision":
				description_final= generate_llama_vision_alternative_text(
					description,
					filename,
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
				description_final= generate_internvl_alternative_text(
					description,
					filename, 
					model, 
					tokenizer,
					temperature=args.temperature,
					resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast	
				)
			else:
				description_final= generate_internvl_alternative_text(
					description,
					filename, 
					model, 
					tokenizer,
					temperature=args.temperature,
					resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast	
				)
					
			description_final= description_final.strip('\n')
			logger.info("Generated description for image %s: %s" % (filename, description_final))
			
		a1= {"from": "gpt", "value": description_final}
	
		# ---------------------------------------
		# - Multi-turn questions-answers
		# .......................................
		if args.generate_qa:
		
			# - Define fig caption
			fig_caption= "The image is a radio astronomical image cutout extracted from a larger radio-continuum Stokes-I map produced by an interferometer telescope. "
			if label=="1C-1P":
				fig_caption+= "The image is centred and zoomed on a radio source of morphological class 1C-1P, consisting in a single-island with only one flux intensity peak. "
			elif label=="1C-2P":
				fig_caption+= "The image is centred and zoomed on a radio source of morphological class 1C-2P, consisting in a single-island with two flux intensity peaks. "
			elif label=="1C-3P":
				fig_caption+= "The image is centred and zoomed on a radio source of morphological class 1C-3P, consisting in a single-island with three flux intensity peaks. "
			elif label=="2C-2P":
				fig_caption+= "The image is centred and zoomed on a radio source of morphological class 2C-2P, consisting of two disjoint islands, each hosting a single flux intensity peak. "			
			elif label=="2C-3P":
				fig_caption+= "The image is centred and zoomed on a radio source of morphological class 2C-3P, consisting of two disjoint islands, where one has a single flux intensity peak and the other one has two intensity peaks. "
			elif label=="3C-3P":
				fig_caption+= "The image is centred and zoomed on a radio source of morphological class 3C-3P, consisting of three disjoint islands, each hosting a single flux intensity peak. "
		
			# - Define query
			query= context + "\n"
			query+= "Figure caption: " + fig_caption + "\n\n"
			query+= glossary + "\n"
			query+= task + "\n"
			query+= task_requirements
			
			logger.info("Generating Q&A from description of image %s: %s" % (filename, fig_caption))
			logger.debug("--> Query: %s" % (query))
		
			response= ""
			if args.model_type=="llama-vision":
				response= run_llama_vision_model_query(
					query,
					filename,
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
					filename, 
					query,
					resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast,
					do_sample=args.do_sample,
					temperature=args.temperature,
					verbose=False
				)
			else:
				response= run_internvl_model_query(
					model,
					tokenizer,
					filename, 
					query,
					resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast,
					do_sample=args.do_sample,
					temperature=args.temperature,
					verbose=False
				)
					
			# - Parse model response
			logger.info("Generated multi-turn Q&A for image %s: %s" % (filename, response))
			response_str= clean_json_string(response.rstrip())
				
			try:
				responses = json.loads(response_str)
			except Exception as e:
				logger.warning("Failed to convert json string response (%s) for image %s to dict list (err=%s), skip image ..." % (response_str, filename, str(e)))
				continue
			
			# - Extract question-answer pairs
			parsed_questions= []
			parsed_answers= []
			generated_qas= []
				
			if not isinstance(responses,list):
				logger.warning("Converted json string response (%s) for image %s is not a dict list, skip image ..." % (response_str, filename))
				continue
					
			if not responses:
				logger.warning("Converted json string response (%s) for image %s is an empty dict list, skip image ..." % (response_str, filename))
				continue
				
			for index, qaentry in enumerate(responses):
				if not isinstance(qaentry, dict):
					logger.warning("Read item of Q&A list for image %s is not a dict, skip Q&A ..." % (filename))
					continue
					
				if 'question' not in qaentry or 'answer' not in qaentry:
					logger.warning("Read item of Q&A list for image %s do not have question/answer keys, skip Q&A ..." % (filename))
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
				generated_qas.extend([q_curr,a_curr])
					
			logger.info("--> #%d Q&A entries generated and parsed for image %s ..." % (len(parsed_questions), filename))
							
							
		# - Add all messages to collection
		#conversations= [
		#	q1, a1
		#]
		conversations= []
		if args.add_image_description:
			conversations.extend([q1,a1])
		
		if args.generate_qa:
			conversations.extend(generated_qas)
			
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
