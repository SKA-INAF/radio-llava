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
	parser.add_argument('--add_default_qa', dest='add_default_qa', action='store_true', help='Add default image Q&A (default=false)')	
	parser.set_defaults(add_default_qa=False)
	parser.add_argument('--add_image_description', dest='add_image_description', action='store_true', help='Add image description in the dataset (default=false)')	
	parser.set_defaults(add_image_description=False)
	
	# - Model options
	parser.add_argument('-model','--model', dest='model', required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help='LLAMA model used to generate variations') 
	parser.add_argument('-model_type','--model_type', dest='model_type', required=False, default="llama", type=str, help='Model to be used {llama, llama-vision}') 
	parser.add_argument('-model_name','--model_name', dest='model_name', required=False, type=str, default="", help='InternVL pretrained model name (e.g. InternVL2_5-1B, ...). This is needed for split device_map.') 
	parser.add_argument('-device_map','--device_map', dest='device_map', required=False, default="auto", type=str, help='Device map used when loading model {auto,split}') 
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
	
	classification_msg_header= "Consider these morphological classes of radio sources, defined as follows: \n EXTENDED: This class comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components; \n DIFFUSE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology; \n "
	
	classification_msg_list= [
		classification_msg_header + "Which of these morphological classes do you see in the image?\n EXTENDED\n DIFFUSE \n Please report the identified classes separated by commas. Report just NONE if the above classes are not present in the image.",
		classification_msg_header + "Report which of these radio source morphologies are present in the image?\n EXTENDED\n DIFFUSE \n Please report only the identified classes separated by commas. Report just NONE if the above classes are not present in the image.",
		classification_msg_header + "Identify the morphological classes of the radio sources contained in the image among the following options \n EXTENDED\n DIFFUSE \n Please report all the identified class labels separated by commas. Report just NONE if the above classes are not present in the image."
	]
	
	galaxy_msg_list= [
		"Do you see any likely radio galaxy with an extended morphology in the image? Answer concisely: Yes or No.",
		"Does the image contain sources with a morphology resembling that of an extended radio galaxy? Answer concisely: Yes or No.",
		"Do you see any potential extended radio galaxy in the presented image? Answer concisely: Yes or No."
	]
	
	context= "## Context: You are an AI assistant specialized in radio astronomical topics. You are given an input image from a scientific research paper along with its corresponding text description (Figure caption) provided below: \n"
	
	glossary= "## Glossary:\n"
	glossary+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level.\n"
	glossary+= "- COMPACT SOURCE: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape. \n"
	glossary+= "- EXTENDED SOURCE: This morphological class of radio sources comprises either single-island compact objects with sharp edges, having a morphology and size dissimilar to that of the image synthesised beam (e.g. 10 times larger than the beam size or with elongated shape), or disjoint multi-island objects, where each island can have either a compact or extended morphology and can host single or multiple emission components. \n"
	glossary+= "- DIFFUSE SOURCE: a particular class of single-island extended objects with small angular size (e.g. smaller than few arcminutes), having diffuse edges and a roundish morphology; \n DIFFUSE-LARGE: large-scale (e.g. larger than few arcminutes and covering a large portion of the image) diffuse object with irregular shape. \n"
	glossary+= "- RADIO GALAXY: a type of active galaxy that emits an exceptionally large amount of radio waves, often extending beyond its visible structure. These galaxies host an active galactic nucleus (AGN), powered by a supermassive black hole (SMBH) at their center, which fuels the production of powerful radio jets and lobes. \n"
	glossary+= "- HII REGION: clouds of ionized hydrogen found in interstellar space, typically surrounding young, hot, massive O- and B-type stars. The high-energy ultraviolet (UV) radiation from these stars ionizes the surrounding hydrogen gas, creating a luminous nebula.\n"
	glossary+= "- SUPERNOVA REMNANT (SNR): expanding, shock-heated shell of gas and dust left behind after a supernova explosion. It consists of material ejected from the explosion, along with swept-up interstellar medium, and is often characterized by complex filamentary structures and strong emission across multiple wavelengths. \n"
	glossary+= "- PLANETARY NEBULA (PN): shell of ionized gas ejected by a low- to intermediate-mass star (typically 1–8 solar masses) during the late stages of its evolution. The exposed hot stellar core emits ultraviolet (UV) radiation, ionizing the expelled gas and causing it to glow.\n"
	glossary+= "- YOUNG STELLAR OBJECT (YSO): forming star in its early evolutionary stages, still accreting material from its surrounding molecular cloud. YSOs are embedded in dusty envelopes and protoplanetary disks and are often detected through their infrared excess due to heated dust and gas. \n"
	glossary+= "- PULSAR: highly magnetized, rapidly rotating neutron star that emits beams of electromagnetic radiation from its magnetic poles. These beams sweep across space like a lighthouse, producing periodic pulses of radio, X-ray, or gamma-ray emission as observed from Earth. \n"
	glossary+= "- LUMINOUS BLUE VARIABLE (LBV): massive, evolved star that experiences dramatic variations in brightness, temperature, and mass loss due to intense stellar instability. These stars are among the most luminous and massive in the universe, typically exceeding 50 solar masses. LBVs undergo episodic eruptions, during which they shed large amounts of material into the surrounding space. \n"
	glossary+= "- WOLF-RAYET (WR) STAR: massive, evolved star that has lost its outer hydrogen envelope due to intense stellar winds, exposing its hot, helium-rich core. These stars are characterized by their high temperatures (25,000–200,000 K), extreme mass loss rates, and broad emission lines in their spectra, caused by fast-moving stellar winds. WR stars represent a late stage in the evolution of massive stars and are potential progenitors of supernovae and gamma-ray bursts.\n"
	glossary+= "- HIGH-MASS X-RAY BINARY STAR (HMXB): binary star system consisting of a massive donor star (typically an O-type, B-type, or Wolf-Rayet star) and a compact object—either a neutron star or a black hole. The compact object accretes material from the massive companion, generating intense X-ray emissions. \n"
	glossary+= "- LOW-MASS X-RAY BINARY STAR (LMXB): binary star system composed of a low-mass donor star (typically less than 1–2 solar masses) and a compact object—either a neutron star or a black hole. The compact object accretes material from the low-mass companion, typically via Roche lobe overflow, producing intense X-ray emissions. \n"
	glossary+= "\n"
	
	task= "## Task: Create multiple precise and self-contained question-answer pairs about the input image using the provided image, context and caption text description. For the question-answer generation you must precisely follow the task requirements described below: \n"
	
	task_requirements= "## Task requirements: Below are requirements for generating the questions and answers in the conversation: \n"
	task_requirements+= "- Adopt an astronomical scientific style in both question formulation and question answers, following definitions and concepts given in the Glossary. \n"
	task_requirements+= "- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information. \n"
	task_requirements+= "- Do not use phrases like \"mentioned\", \"caption\", \"context\" in the conversation. Instead, refer to the information as being \"in the image\". \n"
	task_requirements+= "- Ensure that questions are diverse and cover a range of visual aspects of the image. \n"
	task_requirements+= "- The conversation should include at least 2 or 3 turns of questions and answers about the visual aspects of the image that fully cover all information reported in the provided caption. \n"
	task_requirements+= "- Answer responsibly, without inventing words or sentences that deviate or distort the original figure context and description meaning. \n"
	task_requirements+= "- Answers should be clear, specific, and provide comprehensive information based on the image and its provided context/description. \n"
	task_requirements+= "- Ensure that each question-answer pair incorporates all necessary context, allowing them to be fully understood on their own without external references. \n"
	task_requirements+= "- Include at least one question to describe in brief the image and its content. \n"
	task_requirements+= "- Include at least one question to classify the astronomical type class of the object shown in the image based on its content. \n"
	task_requirements+= "- Include at least one question to determine what morphological kind of radio sources are present in the image. \n"
	task_requirements+= "- Include at least one question to determine whether extended radio galaxies are present in the image based on its content. \n"
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
		filename= item["filename"]
		labels_smorph= item["smorph"]
		labels_sclass= item["sclass"]
		is_multisland= item["multisland"]
		n_islands= item["nislands"]
		
		# - Check smorph tag
		nlabels_smorph= len(labels_smorph)
		nlabels_sclass= len(labels_sclass)
		label_smorph= labels_smorph[0]
		label_sclass= labels_sclass[0]
		if nlabels_smorph>1:
			print("WARN: More than one smorph label (%s) found, taking the first one ..." % (str(nlabels_smorph)))
		
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
		
		# - Add smorph info
		if label_smorph=="EXTENDED":
			if is_multisland:
				description+= "The image is centred and zoomed on a multi-island extended radio source, consisting of " + str(n_islands) + " islands. "
			else:
				description+= "The image is centred and zoomed on a single-island extended radio source. "
		elif label_smorph=="DIFFUSE":
			if is_multisland:
				description+= "The image is centred and zoomed on a multi-island diffuse radio source, consisting of " + str(n_islands) + " islands. "
			else:
				description+= "The image is centred and zoomed on a single-island diffuse radio source. "
		else:
			if is_multisland:
				description+= "The image is centred and zoomed on a single-island compact radio source, consisting of " + str(n_islands) + " islands. "
			else:
				description+= "The image is centred and zoomed on a multi-island compact radio source. "
		
		description+= "An island is a group of 4-connected pixels with flux above a detection threshold with respect to the sky background level. "
				
		# - Add astro class info
		if nlabels_sclass==1:
			if label_sclass=="GALAXY":
				description+= "The visible radio source was classified as an extended radio galaxy candidate on the basis of its radio morphology. "
			elif label_sclass=="SNR":
				description+= "The visible radio source was classified as a supernova remnant (SNR). "
			elif label_sclass=="HII":
				description+= "The visible radio source was classified as a HII region. "
			elif label_sclass=="PN":
				description+= "The visible radio source was classified as a planetary nebula (PN). "
			elif label_sclass=="PULSAR":
				description+= "The visible radio source was classified as a pulsar. "
			elif label_sclass=="LBV":
				description+= "The visible radio source was classified as a luminous blue variable (LBV) star. "
			elif label_sclass=="WR":
				description+= "The visible radio source was classified as a Wolf–Rayet (WR) star. "
			elif label_sclass=="STAR":
				description+= "The visible radio source was classified as a generic radio star. "
			elif label_sclass=="LMXB":
				description+= "The visible radio source was classified as a low-mass X-ray binary (LMXB) star. "
			elif label_sclass=="HMXB":
				description+= "The visible radio source was classified as a high-mass X-ray binary (LMXB) star. "
			elif label_sclass=="YSO":
				description+= "The visible radio source was classified as a young stellar object (YSO). "
		else:
			labels_str= ','.join(labels_sclass)
			description+= "The visible radio source has multiple classification labels (" + labels_str + "), and thus its classification is not unambiguous. "
		
		# - Generate final description
		description_final= description
		
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
		# - Image source morphology classification
		# .......................................		
		q2= {"from": "human", "value": random.choice(classification_msg_list)}
	
		visible_classes= []
		if label_smorph=="EXTENDED":
			visible_classes.append("EXTENDED")
		if label_smorph=="DIFFUSE":
			visible_classes.append("DIFFUSE")
		
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
		if 'GALAXY' in labels_sclass:
			response= "Yes"
			
		a3= {"from": "gpt", "value": response}
	
	
		# ---------------------------------------
		# - Multi-turn questions-answers
		# .......................................
		if args.generate_qa:
		
			# - Define fig caption
			fig_caption= "The image is a radio astronomical image cutout extracted from a larger radio-continuum Stokes-I map produced by an interferometer telescope."
			
			if label_smorph=="EXTENDED":
				if is_multisland:
					fig_caption+= "The image is centred and zoomed on a multi-island extended radio source, consisting of " + str(n_islands) + " islands. "
				else:
					fig_caption+= "The image is centred and zoomed on a single-island extended radio source. "
			elif label_smorph=="DIFFUSE":
				if is_multisland:
					fig_caption+= "The image is centred and zoomed on a multi-island diffuse radio source, consisting of " + str(n_islands) + " islands. "
				else:
					fig_caption+= "The image is centred and zoomed on a single-island diffuse radio source. "
			else:
				if is_multisland:
					fig_caption+= "The image is centred and zoomed on a single-island compact radio source, consisting of " + str(n_islands) + " islands. "
				else:
					fig_caption+= "The image is centred and zoomed on a multi-island compact radio source. "
				
			if nlabels_sclass==1:
				if label_sclass=="GALAXY":
					fig_caption+= "The visible radio source was classified as an extended radio galaxy candidate on the basis of its radio morphology. "
				elif label_sclass=="SNR":
					fig_caption+= "The visible radio source was classified as a supernova remnant (SNR). "
				elif label_sclass=="HII":
					fig_caption+= "The visible radio source was classified as a HII region. "
				elif label_sclass=="PN":
					fig_caption+= "The visible radio source was classified as a planetary nebula (PN). "
				elif label_sclass=="PULSAR":
					fig_caption+= "The visible radio source was classified as a pulsar. "
				elif label_sclass=="LBV":
					fig_caption+= "The visible radio source was classified as a luminous blue variable (LBV) star. "
				elif label_sclass=="WR":
					fig_caption+= "The visible radio source was classified as a Wolf–Rayet (WR) star. "
				elif label_sclass=="STAR":
					fig_caption+= "The visible radio source was classified as a generic radio star. "
				elif label_sclass=="LMXB":
					fig_caption+= "The visible radio source was classified as a low-mass X-ray binary (LMXB) star. "
				elif label_sclass=="HMXB":
					fig_caption+= "The visible radio source was classified as a high-mass X-ray binary (LMXB) star. "
				elif label_sclass=="YSO":
					fig_caption+= "The visible radio source was classified as a young stellar object (YSO). "
			else:
				labels_str= ','.join(labels_sclass)
				fig_caption+= "The visible radio source has multiple classification labels (" + labels_str + "), and thus its classification is not unambiguous. "	
				
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
		#	q1, a1,
		#	q2, a2,
		#	q3, a3
		#]
		conversations= []
		if args.add_image_description:
			conversations.extend([q1,a1])
		
		if args.add_default_qa:
			conversations.extend(
				[
					q2, a2, 
					q3, a3
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
