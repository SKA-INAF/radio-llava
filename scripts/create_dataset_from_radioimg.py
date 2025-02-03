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
	
	# - Model options
	parser.add_argument('--add_image_description', dest='add_image_description', action='store_true', help='Add image description in the dataset (default=false)')	
	parser.set_defaults(add_image_description=False)
	parser.add_argument('--add_default_qa', dest='add_default_qa', action='store_true', help='Add default image Q&A (default=false)')	
	parser.set_defaults(add_default_qa=False)
	parser.add_argument('--generate_text_variations', dest='generate_text_variations', action='store_true', help='Generate text variations using LLAMA model (default=false)')	
	parser.set_defaults(generate_text_variations=False)
	parser.add_argument('--generate_qa', dest='generate_qa', action='store_true', help='Add image generated question-answer in the dataset (default=false)')	
	parser.set_defaults(generate_qa=False)
	
	parser.add_argument('-model','--model', dest='model', required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help='LLAMA model used to generate variations') 
	parser.add_argument('-model_type','--model_type', dest='model_type', required=False, default="llama", type=str, help='Model to be used {llama, llama-vision, internvl}') 
	parser.add_argument('-model_name','--model_name', dest='model_name', required=False, type=str, default="", help='InternVL pretrained model name (e.g. InternVL2_5-1B, ...). This is needed for split device_map.') 
	parser.add_argument('-device_map','--device_map', dest='device_map', required=False, default="auto", type=str, help='Device map used when loading model') 
	parser.add_argument('-max_new_tokens','--max_new_tokens', dest='max_new_tokens', required=False, default=1024, type=int, help='The max number of tokens to be generated') 
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
	
	classification_msg_header= "Consider these morphological classes of radio sources, defined as follows: \n EXTENDED: This class comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components; \n DIFFUSE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology; \n DIFFUSE-LARGE: large-scale (e.g. larger than few arcminutes and covering a large portion of the image) diffuse object with irregular shape.\n "
	
	classification_msg_list= [
		classification_msg_header + "Which of these morphological classes do you see in the image?\n EXTENDED\n DIFFUSE\n DIFFUSE-LARGE \n Please report the identified classes separated by commas. Report just NONE if the above classes are not present in the image.",
		classification_msg_header + "Report which of these radio source morphologies are present in the image?\n EXTENDED\n DIFFUSE\n DIFFUSE-LARGE \n Please report only the identified classes separated by commas. Report just NONE if the above classes are not present in the image.",
		classification_msg_header + "Identify the morphological classes of the radio sources contained in the image among the following options \n EXTENDED\n DIFFUSE\n DIFFUSE-LARGE \n Please report all the identified class labels separated by commas. Report just NONE if the above classes are not present in the image."
	]
	
	galaxy_msg_list= [
		"Do you see any likely radio galaxy with an extended morphology in the image? Answer concisely: Yes or No.",
		"Does the image contain sources with a morphology resembling that of an extended radio galaxy? Answer concisely: Yes or No.",
		"Do you see any potential extended radio galaxy in the presented image? Answer concisely: Yes or No."
	]
	
	artefact_msg_list= [
		"Does the image contain imaging artefacts with a ring pattern around any visible radio source? Answer concisely: Yes or No.",
		"Do you see any imaging artefact around bright sources in the presented image? Answer concisely: Yes or No.",
		"Please report whether the given image contains imaging artefacts or sidelobes around bright compact sources. Answer concisely: Yes or No."
	]

	border_msg_list= [
		"Does the image contain empty pixels at the border? Answer concisely: Yes or No.",
		"Is there any blank pixel region at the edges of the presented image? Answer concisely: Yes or No.",
		"Please report whether the given image has blank (zero or NaN) pixels at the border. Answer concisely: Yes or No."
	]
	
	anomaly_msg_list= [
		"Is the image content ordinary or peculiar in terms of contained objects? ",
		"Do you see any radio source with peculiar morphology in the presented image? ",
		"Please report if the given image contains any radio source with an anomalous or peculiar morphology. "
	]
	
	anomaly_class_msg_header= "Consider this radio image peculiarity classes, defined as follows: \n ORDINARY: image containing only point-like or slightly-resolved compact radio sources superimposed over the sky background or imaging artefact patterns; \n COMPLEX: image containing one or more radio sources with extended or diffuse morphology; \n PECULIAR: image containing one or more radio sources with anomalous or peculiar extended morphology, often having diffuse edges, complex irregular shapes, covering a large portion of the image.\n "

	anomaly_class_msg_list= [
		anomaly_class_msg_header + "Identify and report the peculiarity class of the given image.",
		anomaly_class_msg_header + "Could you determine and report the peculiarity class of the input image?",
		anomaly_class_msg_header + "Can you identify which peculiarity class the presented image belongs to?"
	]
	
	context= "## Context: You are an AI assistant specialized in radio astronomical topics. You are given an input image from a scientific research paper along with its corresponding text description (Figure caption) provided below: \n"
	
	glossary= "## Glossary:\n"
	glossary+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level.\n"
	glossary+= "- COMPACT SOURCE: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape. \n"
	glossary+= "- EXTENDED SOURCE: This morphological class of radio sources comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components. \n"
	glossary+= "- DIFFUSE SOURCE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology; \n DIFFUSE-LARGE: large-scale (e.g. larger than few arcminutes and covering a large portion of the image) diffuse object with irregular shape. \n"
	glossary+= "- ARTEFACTS or SIDELOBES: noise patterns with a ring-like or elongated morphology, typically found around bright compact sources in radio images. They are often mistaken for real radio sources but are actually spurious. These patterns arise from imperfections in the imaging processing stage of radio data. \n"
	glossary+= "- RADIO GALAXY: a type of active galaxy that emits an exceptionally large amount of radio waves, often extending beyond its visible structure. These galaxies host an active galactic nucleus (AGN), powered by a supermassive black hole (SMBH) at their center, which fuels the production of powerful radio jets and lobes. \n"
	glossary+= "- ORDINARY IMAGE: image containing only point-like or slightly-resolved compact radio sources superimposed over the sky background or imaging artefact patterns. \n"
	glossary+= "- COMPLEX IMAGE: image containing one or more radio sources with extended or diffuse morphology. \n"
	glossary+= "- PECULIAR IMAGE: image containing one or more radio sources with anomalous or peculiar extended morphology, often having diffuse edges, complex irregular shapes, covering a large portion of the image. \n"
	glossary+= "\n"
	
	task= "## Task: Create multiple precise and self-contained question-answer pairs about the input image using the provided image, context and caption text description. For the question-answer generation you must precisely follow the task requirements described below: \n"
	
	task_requirements= "## Task requirements: Below are requirements for generating the questions and answers in the conversation: \n"
	task_requirements+= "- Adopt an astronomical scientific style in both question formulation and question answers, following definitions and concepts given in the Glossary. \n"
	task_requirements+= "- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information. \n"
	task_requirements+= "- Do not use phrases like \"mentioned\", \"caption\", \"context\" in the conversation. Instead, refer to the information as being \"in the image\". \n"
	task_requirements+= "- Ensure that questions are diverse and cover a range of visual aspects of the image. \n"
	task_requirements+= "- The conversation should include at least 5 turns of questions and answers about the visual aspects of the image that fully cover all information reported in the provided caption. \n"
	task_requirements+= "- Answer responsibly, without inventing words or sentences that deviate or distort the original figure context and description meaning. \n"
	task_requirements+= "- Answers should be clear, specific, and provide comprehensive information based on the image and its provided context/description. \n"
	task_requirements+= "- Ensure that each question-answer pair incorporates all necessary context, allowing them to be fully understood on their own without external references. \n"
	task_requirements+= "- Include at least one question to classify or identify the complexity/peculiarity/anomaly level of the image based on its content. \n"
	task_requirements+= "- Include at least one question to determine what morphological kind of radio sources are present in the image. \n"
	task_requirements+= "- Include at least one question to determine whether extended radio galaxies or artifacts are present in the image based on its content. \n"
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
		class_ids= item["id"]
		labels= item["label"]
		nlabels= len(labels)
		is_complex= ("EXTENDED" in labels) or ("DIFFUSE" in labels) or ("RADIO-GALAXY" in labels)
		is_wtf= 'WTF' in labels
		
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
		continue_text= False
		if nlabels==1:
			if 'BACKGROUND' in labels:
				description+= "The image contains for a large fraction only sky background noise. "	
			if 'COMPACT' in labels:
				description+= "The image only contains point-like or compact radio sources superimposed over the sky background noise. "
		else:
		
			if 'COMPACT' in labels:
				description+= "The image contains various point-like or compact radio sources superimposed over the sky background noise. "
				continue_text= True
		
			if 'EXTENDED' in labels or 'RADIO-GALAXY' in labels:
				if continue_text:	
					description+= "It also contains one or more extended radio sources. "
				else: 
					description+= "The image contains one or more extended radio sources. "
				if 'RADIO-GALAXY' in labels:
					description+= "Some of them are likely extended radio galaxies. "
				continue_text= True
		
			if 'DIFFUSE' in labels:
				if continue_text:
					description+= "The image also contains roundish diffuse radio sources. "
				else:
					description+= "The image contains roundish diffuse radio sources. "
				continue_text= True
			
			if 'DIFFUSE-LARGE' in labels:
				if continue_text:
					description+= "A large area of diffuse emission is also visible. "
				else:
					description+= "A large area of diffuse emission is visible. "	
				continue_text= True
			
			if 'ARTEFACT' in labels:
				description+= "Some radio sources present in the image are poorly imaged and surrounded by imaging artefacts having a ring pattern. "
			if 'FILAMENT' in labels:
				description+= "Some filamentary structures are present in the image. "
			if 'BORDER' in labels:
				description+= "This image was likely extracted near to mosaic borders as a fraction of the image is empty (NaN or zero pixels). "
			if 'MOSAICING' in labels:
				description+= "The image include residual mosaicking artefact patterns with diagonal line orientation. "
		
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
				
			description_final= description_final.strip('\n')
			print("description (LLAMA generated): ", description_final)
			
		a1= {"from": "gpt", "value": description_final}
	
		# ---------------------------------------
		# - Image source morphology classification
		# .......................................		
		q2= {"from": "human", "value": random.choice(classification_msg_list)}
	
		visible_classes= []
		if 'RADIO-GALAXY' in labels or 'EXTENDED' in labels:
			visible_classes.append("EXTENDED")
		if 'DIFFUSE' in labels:
			visible_classes.append("DIFFUSE")
		if 'DIFFUSE-LARGE' in labels:
			visible_classes.append("DIFFUSE-LARGE")
			
		if visible_classes:
			visible_classes_str= ','.join(visible_classes)
		else:
			visible_classes_str= "NONE"	
		
		a2= {"from": "gpt", "value": visible_classes_str}
	
	
		# ---------------------------------------
		# - Radio galaxy question
		# .......................................		
		q3= {"from": "human", "value": random.choice(galaxy_msg_list)} 
	
		response= "No"
		if 'RADIO-GALAXY' in labels:
			response= "Yes"
			
		a3= {"from": "gpt", "value": response}
	
		# ---------------------------------------
		# - Artefact question
		# .......................................		
		q4= {"from": "human", "value": random.choice(artefact_msg_list)} 
	
		response= "No"
		if 'ARTEFACT' in labels:
			response= "Yes"
	
		a4= {"from": "gpt", "value": response}
	
		# ---------------------------------------
		# - Image border question
		# .......................................
		q5= {"from": "human", "value": random.choice(border_msg_list)} 
	
		response= "No"
		if 'BORDER' in labels:
			response= "Yes"
	
		a5= {"from": "gpt", "value": response}
	
		# ---------------------------------------
		# - Anomaly question
		# .......................................
		q6= {"from": "human", "value": random.choice(anomaly_msg_list)} 
	
		response= ""
		if is_complex:
			response= "The image contains radio sources with an extended or diffuse morphology that could be relevant or interesting for the user, depending on the analysis case or field of study."
			if is_wtf:
				response+= " Some of these radio sources have a very peculiar morphology."
		else:
			if is_wtf:
				response= "The image contains radio sources with a very peculiar morphology."			
			else:
				if nlabels==1 and ('BACKGROUND' in labels or 'COMPACT' in labels):
					response= "The image is very ordinary as it does contain only compact or point-like radio sources superimposed over the sky background. "
				else:
					response= "The image is ordinary and does not contain radio sources with a particular morphological structure. "
	
		description_anomaly= response
		
		a6= {"from": "gpt", "value": description_anomaly}
		
		# ---------------------------------------
		# - Anomaly class question
		# .......................................
		q7= {"from": "human", "value": random.choice(anomaly_class_msg_list)} 
	
		response= "ORDINARY"
		if is_wtf:
			response= "PECULIAR"
		else:
			if is_complex:
				response= "COMPLEX"
		
		a7= {"from": "gpt", "value": response}
		
		# ---------------------------------------
		# - Multi-turn questions-answers
		# .......................................
		if args.generate_qa:
		
			# - Define fig caption
			fig_caption= "The image is a radio astronomical image cutout extracted from a larger radio-continuum Stokes-I map produced by an interferometer telescope. "
			
			if 'BACKGROUND' in labels:
				fig_caption+= "The image contains for a large fraction only sky background noise. "
			if 'COMPACT' in labels:
				fig_caption+= "The image contains various point-like or compact radio sources superimposed over the sky background noise. "
			if 'EXTENDED' in labels:
				fig_caption+= "The image contains one or more extended radio sources. "
			else:
				fig_caption+= "The image does not contain extended radio sources. "
			
			if 'RADIO-GALAXY' in labels:
				fig_caption+= "The image contains candidate radio galaxies with extended morphology. "
			else:
				fig_caption+= "The image does not contain candidate radio galaxies with extended morphology. "
			
			if 'DIFFUSE' in labels:
				fig_caption+= "The image contains roundish diffuse radio sources. "
			else:
				fig_caption+= "The image does not contain roundish diffuse radio sources. "
					
			if 'DIFFUSE-LARGE' in labels:
				fig_caption+= "A large area of diffuse emission is visible in the image. "
			else:	
				fig_caption+= "The image does not present areas of large diffuse emission. "
				
			if 'ARTEFACT' in labels:
				fig_caption+= "The image contains imaging artefacts, e.g. some of the radio sources present in the image are poorly imaged and surrounded by imaging artefacts having a ring pattern. "
			else:
				fig_caption+= "The image does not contain imaging artefacts. "
			
			if 'FILAMENT' in labels:
				fig_caption+= "Some filamentary structures are present in the image. "
			else:
				fig_caption+= "No filamentary structures are present in the image. "
				
			if 'BORDER' in labels:
				fig_caption+= "The image was likely extracted near to mosaic borders as a fraction of the image is empty (NaN or zero pixels). "
			else:
				fig_caption+=	"The image does not contain regions with empty or NaN pixels. "
				
			if 'MOSAICING' in labels:
				fig_caption+= "The image contains residual mosaicking artefact patterns with a diagonal line orientation. "	
			else:
				fig_caption+= "The image does not contain mosaicking artefact patterns with a diagonal line orientation. "
		
			if is_complex:
				fig_caption+= "The image can be considered complex, as it contains radio sources with extended or diffuse morphology that may be relevant or interesting to the user, depending on the analysis context or field of study. "
				if is_wtf:
					fig_caption+= " Some of these radio sources exhibit highly peculiar morphologies, making the image itself distinctive."
			else:
				if is_wtf:
					fig_caption+= "The image contains radio sources with a very peculiar morphology. "			
				else:
					if nlabels==1 and ('BACKGROUND' in labels or 'COMPACT' in labels):
						fig_caption+= "The image is very ordinary as it does contain only compact or point-like radio sources superimposed over the sky background. "
					else:
						fig_caption+= "The image is ordinary and does not contain radio sources with a particular morphological structure. "		
		
			# - Define query
			#fig_caption= description_final
			#fig_caption+= description_anomaly
			
			query= context + "\n"
			query+= "Figure caption: " + fig_caption + "\n\n"
			query+= glossary + "\n"
			query+= task + "\n"
			query+= task_requirements
			
			logger.info("Generating Q&A from description of image %s: %s" % (filename, fig_caption))
			logger.info("--> Query: %s" % (query))
		
			
			response= ""
			if args.model_type=="llama-vision":
				response= run_llama_vision_model_query(
					query,
					filename,
					model,
					processor,
					do_sample=False,
					temperature=args.temperature,
					max_new_tokens=1024,
					top_p=1.0,
					top_k=20,
					penalty=1.2,
					resize=False, resize_size=args.imgsize,
					zscale=False, contrast=0.25,
					verbose=False
				)
			elif args.model_type=="internvl":
				response= run_internvl_model_query(
					model,
					tokenizer,
					filename, 
					query,
					resize_size=args.imgsize,
					zscale=False, contrast=0.25,
					do_sample=False,
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
					zscale=False, contrast=0.25,
					do_sample=False,
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
		#	q1, a1,
		#	q2, a2,
		#	q3, a3,
		#	q4, a4,
		#	q5, a5,
		#	q6, a6,
		#	q7, a7
		#]
		conversations= []
		if args.add_image_description:
			conversations.extend([q1,a1])
			
		if args.add_default_qa:
			conversations.extend(
				[
					q2, a2, 
					q3, a3, 
					q4, a4, 
					q5, a5,
					q6, a6,
					q7, a7
				]
			)
		
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
	
