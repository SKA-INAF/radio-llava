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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input data filelist with images') 
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, default=-1, type=int, help='Max number of processed images') 
	
	# - Run options
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
	model_id= args.model
	
	id2label= {
		0: "ARTEFACT",
  	1: "COMPACT",
  	2: "EXTENDED",
  	3: "EXTENDED-MULTISLAND",
  	4: "FLAGGED"
	}

		
	#===========================
	#==   READ DATA
	#===========================	
	# - Read datalist
	logger.info("Read datalist %s ..." % (args.inputfile))
	with open(inputfile, 'r', encoding="ascii") as f:
		lines = f.readlines()

	datalist= []
	for line in lines:
		line = line.replace("\n", "")
		datalist.append(line)
		
	#===========================
	#==   LOAD LLAMA MODEL
	#===========================
	model= None
	tokenizer= None
	processor= None
	logger.info("Loading model %s ..." % (model_id))
	if args.model_type=="llama":
		model, tokenizer= load_llama_model(model_id, args.device_map)
	elif args.model_type=="llama-vision":
		model, processor= load_llama_vision_model(model_id)
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
	
	context= "## Context: You are an AI assistant specialized in radio astronomical topics. You are given an input image from a scientific research paper along with its corresponding text description (Figure caption) provided below: \n"
	
	glossary= "## Glossary:\n"
	glossary+= "- SOURCE ISLAND: A group of 4-connected pixels in a radio image under analysis with intensity above a detection threshold with respect to the sky background level. The terms 'island' and 'component' are sometimes used interchangeably, but an 'island' is defined purely by pixel connectivity above a noise threshold, and may contain one or multiple source components. Example: A radio galaxy with extended lobes may appear as one large source island, encompassing multiple structures (core, jets, lobes).\n"
	glossary+= "- COMPACT SOURCE: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape of the image. \n"
	glossary+= "- EXTENDED SOURCE: single-island radio sources with extended morphology, eventually hosting one or more blended components, with some deviating from the synthesized beam shape. \n"
	glossary+= "- EXTENDED MULTI-ISLAND: radio sources with an extended morphology, consisting of more than one disjoint island, where each island can have either a compact or extended morphology and can host single or multiple emission components. \n"
	glossary+= "- SPURIOUS SOURCE: spurious/fake sources, due to artefacts introduced in the radio image by the imaging process, having a ring-like or elongated compact morphology\n"
	glossary+= "- FLAGGED SOURCE: single-island radio sources, with compact or extended morphology, that are poorly imaged and largely overlapping with close imaging artefacts.\n"
	glossary+= "\n"
		
	task= "## Task: Create multiple precise and self-contained question-answer pairs about the input image using the provided image, context and caption text description. For the question-answer generation you must precisely follow the task requirements described below: \n"
	
	task_requirements= "## Task requirements: Below are requirements for generating the questions and answers in the conversation: \n"
	task_requirements+= "- Adopt an astronomical scientific style in both question formulation and question answers, following definitions and concepts given in the Glossary. \n"
	task_requirements+= "- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information. \n"
	task_requirements+= "- Do not use phrases like \"mentioned\", \"caption\", \"context\" in the conversation. Instead, refer to the information as being \"in the image\". \n"
	task_requirements+= "- Ensure that questions are diverse and cover a range of visual aspects of the image. \n"
	task_requirements+= "- The conversation should include at least 4 or 5 turns of questions and answers about the visual aspects of the image that fully cover all information reported in the provided caption. \n"
	task_requirements+= "- Answer responsibly, without inventing words or sentences that deviate or distort the original figure context and description meaning. \n"
	task_requirements+= "- Answers should be clear, specific, and provide comprehensive information based on the image and its provided context/description. \n"
	task_requirements+= "- Ensure that each question-answer pair incorporates all necessary context, allowing them to be fully understood on their own without external references. \n"
	task_requirements+= "- Include at least one question for each source class (SPURIOUS, COMPACT, EXTENDED, EXTENDED-MULTISLAND, FLAGGED) to determine whether that radio source category is present in the image.\n"
	task_requirements+= "- Include at least one question for each source class (SPURIOUS, COMPACT, EXTENDED, EXTENDED-MULTISLAND, FLAGGED) to determine the bounding box positions of all radio sources of those classes present in the image. Example: 'Please provide the bounding box coordinates of all extended multi-island sources present in the image.' \n"
	task_requirements+= "- Include at least one question to assess the presence of artifacts or likely spurious sources in the image based on its content. \n"
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
	

	for idx, filename in enumerate(datalist):
		# - Check stop condition
		if args.nmax!=-1 and idx>=args.nmax:
			logger.info("Stop loop condition reached (%d), as #%d entries were processed..." % (args.nmax, idx))
			break
	
		uuid = shortuuid.uuid()
		
		# - Find annotation data
		image_name= os.path.basename(filename)
		dir_ann= os.path.dirname(filename).replace("images","labels")
		filename_ann= os.path.join(dir_ann, image_name.replace(".png",".txt"))
		
		# - Process annotation data
		t= ascii.read(filename_ann)
		
		obj_info= {
			"ARTEFACT": {
				"count": 0, 
				"bboxes": [],
				"description": "artefacts introduced in the radio image by the imaging process, having a ring-like or elongated compact morphology"
			},
			"COMPACT": {
				"count": 0, 
				"bboxes": [],
				"description": "single-island isolated point-like or slightly resolved radio sources with well-defined edges, hosting one or more blended components, each with morphology resembling the synthesized beam shape of the image"
			},
			"EXTENDED": {
				"count": 0, 
				"bboxes": [],
				"description": "radio sources with a single-island extended morphology, eventually hosting one or more blended components, with some deviating from the synthesized beam shape"
			},
			"EXTENDED-MULTISLAND": {
				"count": 0, 
				"bboxes": [],
				"description": "radio sources with an extended morphology, consisting of more (point-like or extended) islands, each one hosting one or more blended components"
			},
			"FLAGGED": {
				"count": 0, 
				"bboxes": [],
				"description": "single-island radio sources, with compact or extended morphology, that are poorly imaged and largely overlapping with close imaging artefacts"
			}		
		}
		
		only_compact= True
		has_compact= False
		has_extended= False
		has_extended_multi= False
		has_artefact= False
		has_flagged= False
		
		for item in t:
			class_id= item[0]
			x_center= float(round(item[1], 2)) 
			y_center= float(round(item[2], 2))
			width= float(round(item[3], 2))
			height= float(round(item[4], 2))
			bbox= (x_center, y_center, width, height)
			class_label= id2label[class_id]
			obj_info[class_label]["count"]+= 1
			obj_info[class_label]["bboxes"].append(bbox)
			if class_label=="COMPACT":
				has_compact= True
			if class_label=="EXTENDED":
				has_extended= True
			if class_label=="EXTENDED-MULTISLAND":
				has_extended_multi= True
			if class_label=="EXTENDED" or class_label=="EXTENDED-MULTISLAND" or class_label=="ARTEFACT":
				only_compact= False
			if class_label=="ARTEFACT":
				has_artefact= True
			if class_label=="FLAGGED":
				has_flagged= True
				
		print("--> obj_info")
		print(obj_info)
		
		n_compact= obj_info["COMPACT"]["count"]
		n_ext= obj_info["EXTENDED"]["count"]
		n_extmulti= obj_info["EXTENDED-MULTISLAND"]["count"]
		n_ext_tot= n_ext + n_extmulti
		n_artefact= obj_info["ARTEFACT"]["count"]
		n_flagged= obj_info["FLAGGED"]["count"]
	
		# - Initialize outdict
		outdict = dict()
		outdict['id'] = uuid
		outdict['image'] = filename
	
		# - Fill message conversations
		conversations = []
		
		# ---------------------------------------
		# - Image generated description
		# .......................................
		query= "Consider a radio astronomical image extracted from a radio-continuum survey map produced by an interferemoter telescope. Let's consider these possible classes of radio sources that can be contained in a radio astronomical image: \n COMPACT: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape of the image; \n EXTENDED: radio sources with a single-island extended morphology, eventually hosting one or more blended components, with some deviating from the synthesized beam shape; \n EXTENDED-MULTISLAND: including radio sources with an extended morphology, consisting of more (point-like or extended) islands, each one eventually hosting one or more blended components; \n SPURIOUS: spurious sources, due to artefacts introduced in the radio image by the imaging process, having a ring-like or elongated compact morphology; \n FLAGGED: including single-island radio sources, with compact or extended morphology, that are poorly imaged and largely overlapping with close imaging artefacts. \n Generate a brief description of a radio astronomical image containing the following list of radio source objects, each identified by a class label and normalized bounding box pixel coordinates (x,y,w,h): \n "
		
		#query= "Can you write a brief description of a radio astronomical image containing the following list of radio source objects, each represented by a class label and normalized bounding boxes (x,y,w,h)? "
		#query= "Can you write a brief text that describes a radio astronomical image containing the following list of radio source objects, each represented by a class label and normalized bounding boxes in pixel coordinates (x,y,w,h)? "
		
		if args.model_type=="llama-vision" or args.model_type=="internvl":
			query2= "Generate a brief description of the input radio astronomical image containing the following list of radio source objects: \n"
		else:
			query2= "Generate a brief description of a radio astronomical image containing the following list of radio source objects: \n"
		
		text= ""
		for i in range(n_compact):
			bbox= obj_info["COMPACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			text+= "COMPACT: [" + bbox_str + "] \n"
			
		query+= text
		
			
		text= ""
		for i in range(n_ext):
			bbox= obj_info["EXTENDED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			text+= "EXTENDED: [" + bbox_str + "] \n"
			
		query+= text
			
		text= ""
		for i in range(n_extmulti):
			bbox= obj_info["EXTENDED-MULTISLAND"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			text+= "EXTENDED-MULTISLAND: [" + bbox_str + "] \n"
			
		query+= text
			
		text= ""
		for i in range(n_artefact):
			bbox= obj_info["ARTEFACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			text+= "SPURIOUS: [" + bbox_str + "] \n"
			
		query+= text
			
		text= ""
		for i in range(n_flagged):
			bbox= obj_info["FLAGGED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			text+= "FLAGGED: [" + bbox_str + "] \n"
			
		query+= text
			
		# - Alternative text
		text= str(n_compact) + " compact radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_compact):
			bbox= obj_info["COMPACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_compact-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Compact sources are " + obj_info["COMPACT"]["description"] + ".\n"
		if n_compact>0:
			query2+= text
		
		text= str(n_ext) + " extended radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_ext):
			bbox= obj_info["EXTENDED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_ext-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Extended sources are " + obj_info["EXTENDED"]["description"] + ".\n"
		if n_ext>0:
			query2+= text
		
		text= str(n_extmulti) + " extended-multisland radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_extmulti):
			bbox= obj_info["EXTENDED-MULTISLAND"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_extmulti-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Extended multi-island sources are " + obj_info["EXTENDED-MULTISLAND"]["description"] + ".\n"
		if n_extmulti>0:
			query2+= text
		
		text= str(n_artefact) + " spurious radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_artefact):
			bbox= obj_info["ARTEFACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_artefact-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Spurious sources are " + obj_info["ARTEFACT"]["description"] + ".\n"
		if n_artefact>0:
			query2+= text
		
		text= str(n_flagged) + " flagged radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_flagged):
			bbox= obj_info["FLAGGED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_flagged-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Flagged sources are " + obj_info["FLAGGED"]["description"] + ".\n"
		if n_flagged>0:
			query2+= text
		
		
		# - Define additional prompt requirements
		#prompt_requirements= "Use terms like top/bottom, left/right or image width/height fractions or percentages to report source object positions, rather than their exact bounding box coordinates. Please report just the description text using an astronomical scientific style, without any prefix, preamble or explanation or special characters. "
		prompt_requirements= "Include in the description only the objects given in the above list. Use an astronomical scientific style and terms like top/bottom, left/right or image width/height fractions or percentages to describe the source object positions, rather than their exact bounding box coordinates. Avoid lengthy explanations or preambles and special unicode or ascii characters. "
		
		query+= prompt_requirements
		query2+= prompt_requirements
			
		logger.info("--> Processing image %s " % (filename))
		#print(query)
		print(query2)
		print("")
		
		#print("obj_info")
		#print(obj_info)
		
		if args.add_image_description:
			if args.model_type=="llama":
				gen_description= run_llama_model_query(
					query2, 
					model, tokenizer, 
					do_sample=True,
					temperature=args.temperature,
					max_new_tokens=args.max_new_tokens,
					top_p=args.top_p,
					top_k=args.top_k,
					penalty=args.penalty
				)
			
			elif args.model_type=="llama-vision":
				gen_description= run_llama_vision_model_query(
					query2,
					filename,
					model, processor, 
					do_sample=True,
					temperature=args.temperature,
					max_new_tokens=args.max_new_tokens,
					top_p=args.top_p,
					top_k=args.top_k,
					penalty=args.penalty,
					resize=args.resize, resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast
				)
			elif args.model_type=="internvl":
				gen_description= generate_internvl_alternative_text(
					query2,
					filename, 
					model, 
					tokenizer,
					temperature=args.temperature,
					resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast	
				)
			else:
				gen_description= generate_internvl_alternative_text(
					query2,
					filename, 
					model, 
					tokenizer,
					temperature=args.temperature,
					resize_size=args.imgsize,
					zscale=args.zscale, contrast=args.contrast	
				)
				
			gen_description= gen_description.strip('\n')
		
			#print("description (LLAMA-generated)")
			#print(gen_description)
			#print("")
			logger.info("Generated description for image %s: %s" % (filename, gen_description))
				
			q1= {"from": "human", "value": "<image>\n" + random.choice(description_list)}
			a1= {"from": "gpt", "value": gen_description}
		
		# ---------------------------------------
		# - Source bounding boxes
		# .......................................
		# - Compact
		qbbox_compact= {"from": "human", "value": "Please provide the bounding box coordinates of all compact sources present in the image."}
	
		coords= ''
		for i in range(n_compact):
			bbox= obj_info["COMPACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_compact-1:
				coords+= ', '
			#else:
			#	coords+= '. '
		
		response= coords
		if coords=="":
			response= "The image does not contain compact radio sources."
					
		abbox_compact= {"from": "gpt", "value": response}
		
		# - Extended
		qbbox_extended= {"from": "human", "value": "Please provide the bounding box coordinates of all single-island extended sources present in the image."}
	
		coords= ''
		for i in range(n_ext):
			bbox= obj_info["EXTENDED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_ext-1:
				coords+= ', '
			#else:
			#	coords+= '. '
		
		response= coords
		if coords=="":
			response= "The image does not contain single-island extended radio sources."
					
		abbox_extended= {"from": "gpt", "value": response}
		
		# - Extended-multisland
		qbbox_extendedmulti= {"from": "human", "value": "Please provide the bounding box coordinates of all multi-island extended sources present in the image."}
	
		coords= ''
		for i in range(n_extmulti):
			bbox= obj_info["EXTENDED-MULTISLAND"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_extmulti-1:
				coords+= ', '
			#else:
			#	coords+= '. '
		
		response= coords
		if coords=="":
			response= "The image does not contain multi-island extended radio sources."
					
		abbox_extendedmulti= {"from": "gpt", "value": response}
		
		# - Artefacts
		qbbox_artefact= {"from": "human", "value": "Does the image contain spurious sources or any imaging artefact around bright sources? Please provide their bounding box coordinates."}
	
		coords= ''
		for i in range(n_artefact):
			bbox= obj_info["ARTEFACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_artefact-1:
				coords+= ', '
			#else:
			#	coords+= '. '
		
		response= coords
		if coords=="":
			response= "The image does not contain imaging artefacts."
		else:
			response= "The image contains imaging artefacts. Their bounding box coordinates are: " + coords
		
		abbox_artefact= {"from": "gpt", "value": response}
		
		# - Flagged 
		qbbox_flagged= {"from": "human", "value": "Does the image contain radio sources that are to be flagged as severely contaminated or overlapping with imaging artefacts? Please provide their bounding box coordinates."}
	
		coords= ''
		for i in range(n_flagged):
			bbox= obj_info["FLAGGED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_flagged-1:
				coords+= ', '
			#else:
			#	coords+= '. '
		
		response= coords
		if coords=="":
			response= "The image does not contain radio sources to be flagged."
		else:
			response= "The image contains radio sources that are to be flagged as poorly imaged or significantly overlapping with imaging artefacts. Their bounding box coordinates are: " + coords
		
		abbox_flagged= {"from": "gpt", "value": response}
		
		
		# ---------------------------------------
		# - Multi-turn questions-answers
		# .......................................
		if args.generate_qa:
		
			# - Define fig caption
			fig_caption= "The image is a radio astronomical image cutout extracted from a larger radio-continuum Stokes-I map produced by an interferometer telescope. "
			
			# ----- COMPACT -----
			if n_compact<=0:
				fig_caption+= "The image does not contain compact radio sources. "
			else:
				text= "The image includes " + str(n_compact) + " compact radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
				coords= ''
				for i in range(n_compact):
					bbox= obj_info["COMPACT"]["bboxes"][i]
					bbox_str= (str(bbox)[1:-1])
					coords+= "[" + bbox_str + "]"
					if i!=n_compact-1:
						coords+= ', '
					else:
						coords+= '. '
		
				text+= coords
				fig_caption+= text
				
			# ----- EXTENDED -----
			if n_ext<=0:
				fig_caption+= "The image does not contain extended radio sources. "
			else:
			
				text= "The image includes " + str(n_ext) + " extended radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
				coords= ''
				for i in range(n_ext):
					bbox= obj_info["EXTENDED"]["bboxes"][i]
					bbox_str= (str(bbox)[1:-1])
					coords+= "[" + bbox_str + "]"
					if i!=n_ext-1:
						coords+= ', '
					else:
						coords+= '. '
		
				text+= coords
				fig_caption+= text
				
			# ----- EXTENDED MULTI-ISLAND -----	
			if n_extmulti<=0:
				fig_caption+= "The image does not contain extended multi-island radio sources. "
			else:
				
				text= "The image includes " + str(n_extmulti) + " extended multi-island radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
				coords= ''
				for i in range(n_extmulti):
					bbox= obj_info["EXTENDED-MULTISLAND"]["bboxes"][i]
					bbox_str= (str(bbox)[1:-1])
					coords+= "[" + bbox_str + "]"
					if i!=n_extmulti-1:
						coords+= ', '
					else:
						coords+= '. '
		
				text+= coords
				fig_caption+= text
		
			# ----- SPURIOUS -----
			if n_artefact<=0:
				fig_caption+= "The image does not contain spurious radio sources. "
			else:
			
				text= "The image includes " + str(n_artefact) + " spurious radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
				coords= ''
				for i in range(n_artefact):
					bbox= obj_info["ARTEFACT"]["bboxes"][i]
					bbox_str= (str(bbox)[1:-1])
					coords+= "[" + bbox_str + "]"
					if i!=n_artefact-1:
						coords+= ', '
					else:
						coords+= '. '
		
				text+= coords
				fig_caption+= text
		
			# ----- FLAGGED -----
			if n_flagged<=0:
				fig_caption+= "The image does not contain flagged radio sources. "
			else:
			
				text= "The image includes " + str(n_flagged) + " flagged radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
				coords= ''
				for i in range(n_flagged):
					bbox= obj_info["FLAGGED"]["bboxes"][i]
					bbox_str= (str(bbox)[1:-1])
					coords+= "[" + bbox_str + "]"
					if i!=n_flagged-1:
						coords+= ', '
					else:
						coords+= '. '
		
				text+= coords
				fig_caption+= text
			
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
		#	qbbox_compact, abbox_compact,
		#	qbbox_extended, abbox_extended,
		#	qbbox_extendedmulti, abbox_extendedmulti,
		#	qbbox_artefact, abbox_artefact,
		#	qbbox_flagged, abbox_flagged
		#]
		conversations= []
		
		if args.add_image_description:
			conversations.extend([q1,a1])
		
		if args.add_default_qa:
			conversations.extend(
				[
					qbbox_compact, abbox_compact,
					qbbox_extended, abbox_extended,
					qbbox_extendedmulti, abbox_extendedmulti,
					qbbox_artefact, abbox_artefact,
					qbbox_flagged, abbox_flagged
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
		
