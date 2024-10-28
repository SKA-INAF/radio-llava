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
import numpy as np

# - ASTROPY MODULE
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ZScaleInterval
import astropy.units as u
from astropy.wcs.utils import skycoord_to_pixel

# - DRAW MODULES
import matplotlib.pyplot as plt

## MODULE
from radio_llava.utils import *
from radio_llava.inference_llama import *

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input ann data file (.json)') 
	
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, default=-1, type=int, help='Max number of processed images') 
	
	# - Run options
	parser.add_argument('-model','--model', dest='model', required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help='LLAMA model used to generate variations')
	parser.add_argument('-model_type','--model_type', dest='model_type', required=False, default="llama", type=str, help='Model to be used {llama, llama-vision}') 
	
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
	model_id= args.model
	
	#===========================
	#==   READ ANN DATA
	#===========================	
	# - Read annotation data
	logger.info("Reading ann data %s ..." % (inputfile))
	fp= open(inputfile, "r")
	anndata= json.load(fp)
	logger.info("#%d images in dataset ..." % (len(anndata)))
	
	
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
	
	for idx, item in enumerate(anndata):
		# - Check stop condition
		if args.nmax!=-1 and idx>=args.nmax:
			logger.info("Stop loop condition reached (%d), as #%d entries were processed..." % (args.nmax, idx))
			break
			
		# - Retrieve ann info
		obj_info= {
			"UNKNOWN": {
				"count": 0, 
				"bboxes": [],
				"smorph": [],
				"description": "unclassified source"
			},
			"C": {
				"count": 0, 
				"bboxes": [],
				"smorph": [],
				"description": "single-island isolated point-like or slightly resolved radio sources with well-defined edges, hosting one or more blended components, each with morphology resembling the synthesized beam shape of the image"
			},
			"FR-I": {
				"count": 0, 
				"bboxes": [],
				"smorph": [],
				"description": "radio-loud galaxies characterized by a jet-dominated structure where the radio emissions are strongest close to the galaxy's center and diminish with distance from the core"
			},
			"FR-II": {
				"count": 0, 
				"bboxes": [],
				"smorph": [],
				"description": "radio-loud galaxies characterized by a edge-brightened radio structure, where the radio emissions are more prominent in lobes located far from the galaxy's core, with hotspots at the ends of powerful, well-collimated jets"
			},
			"FR-x": {
				"count": 0, 
				"bboxes": [],
				"smorph": [],
				"description": "radio galaxies with mixed or hybrid morphology, showing characteristics of both FR-I and FR-II galaxy classes"
			},
			"R": {
				"count": 0, 
				"bboxes": [],
				"smorph": [],
				"description": "radio galaxies with single-peak resolved morphology"
			}	
		}
		
		obj_info_caesar= {
			"UNKNOWN": {
				"count": 0, 
				"bboxes": [],
				"description": "unclassified source"
			},
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
		
		filename= item["filename"]
		uuid = shortuuid.uuid()
		objs= item["objs"]
		
		# - Compute flags
		only_compact= True
		has_compact= False
		has_extended= False
		has_extended_multi= False
		has_artefact= False
		has_flagged= False
		has_fri= False
		has_frii= False
		has_frx= False
		has_fr= False
		
		for item in objs:
			
			# - Count caesar labels
			label_caesar= "UNKNOWN"
			bbox_caesar= []
			if "label_caesar" in objs:
				label_caesar= item["label_caesar"]
				bbox_caesar= item["bbox_norm_caesar"]
			
			obj_info_caesar[label_caesar]["count"]+= 1
			obj_info_caesar[label_caesar]["bboxes"].append(bbox_caesar)
			if label_caesar=="COMPACT":
				has_compact= True
			if label_caesar=="EXTENDED":
				has_extended= True
			if label_caesar=="EXTENDED-MULTISLAND":
				has_extended_multi= True
			if label_caesar=="EXTENDED" or label_caesar=="EXTENDED-MULTISLAND" or label_caesar=="ARTEFACT":
				only_compact= False
			if label_caesar=="ARTEFACT":
				has_artefact= True
			if label_caesar=="FLAGGED":
				has_flagged= True
				
			# - Count RG-CAT labels
			label= item["label"]
			bbox= item["bbox_norm"]
			obj_info[label]["count"]+= 1
			obj_info[label]["bboxes"].append(bbox)
			obj_info[label]["smorph"].append(label_caesar)
			
			if label=="FR-I":
				has_fri= True
			if label=="FR-II":
				has_frii= True
			if label=="FR-x":
				has_frx= True
			
				
		print("--> obj_info")
		print(obj_info)
		
		print("--> obj_info_caesar")
		print(obj_info_caesar)
		
		has_fr= (has_fri or has_frii or has_frx)
		n_fri= obj_info_caesar["FR-I"]["count"]
		n_frii= obj_info_caesar["FR-II"]["count"]
		n_frx= obj_info_caesar["FR-x"]["count"]
		
		n_compact= obj_info_caesar["COMPACT"]["count"]
		n_ext= obj_info_caesar["EXTENDED"]["count"]
		n_extmulti= obj_info_caesar["EXTENDED-MULTISLAND"]["count"]
		n_ext_tot= n_ext + n_extmulti
		n_artefact= obj_info_caesar["ARTEFACT"]["count"]
		n_flagged= obj_info_caesar["FLAGGED"]["count"]

		# - Initialize outdict
		outdict = dict()
		outdict['id'] = uuid
		outdict['image'] = filename
	
		# - Fill message conversations
		conversations = []
		
		# ---------------------------------------
		# - Image generated description
		# .......................................
		if args.model_type=="llama-vision":
			query= "Generate a brief description of the input radio astronomical image containing the following list of radio source objects: \n"
		else:
			query= "Generate a brief description of a radio astronomical image containing the following list of radio source objects: \n"
		
		text= str(n_compact) + " compact radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_compact):
			bbox= obj_info_caesar["COMPACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_compact-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Compact sources are " + obj_info_caesar["COMPACT"]["description"] + ".\n"
		if n_compact>0:
			query+= text
		
		text= str(n_ext) + " extended radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_ext):
			bbox= obj_info_caesar["EXTENDED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_ext-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Extended sources are " + obj_info_caesar["EXTENDED"]["description"] + ".\n"
		if n_ext>0:
			query+= text
		
		text= str(n_extmulti) + " extended-multisland radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_extmulti):
			bbox= obj_info_caesar["EXTENDED-MULTISLAND"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_extmulti-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Extended multi-island sources are " + obj_info_caesar["EXTENDED-MULTISLAND"]["description"] + ".\n"
		if n_extmulti>0:
			query+= text
		
		text= str(n_artefact) + " spurious radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_artefact):
			bbox= obj_info_caesar["ARTEFACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_artefact-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Spurious sources are " + obj_info_caesar["ARTEFACT"]["description"] + ".\n"
		if n_artefact>0:
			query+= text
		
		text= str(n_flagged) + " flagged radio sources located at these normalized bounding box pixel coordinates (x,y,w,h): "
		coords= ''
		for i in range(n_flagged):
			bbox= obj_info_caesar["FLAGGED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_flagged-1:
				coords+= ', '
			else:
				coords+= '. '
		
		text+= coords
		text+= "Flagged sources are " + obj_info_caesar["FLAGGED"]["description"] + ".\n"
		if n_flagged>0:
			query+= text
			
		# - Add FR information
		if has_fr:
			#query+= "Some radio sources in the image have been classified as radio galaxy of various morphological classes."
		
			# - FR-I
			text= str(n_fri) + " radio galaxies of FR-I type located at these normalized bounding box pixel coordinates (x,y,w,h): "
			coords= ''
			for i in range(n_fri):
				bbox= obj_info["FR-I"]["bboxes"][i]
				bbox_str= (str(bbox)[1:-1])
				coords+= "[" + bbox_str + "]"
				if i!=n_fri-1:
					coords+= ', '
				else:
					coords+= '. '
		
			text+= coords
			text+= "FR-I type radio sources are " + obj_info["FR-I"]["description"] + ".\n"
			if n_fri>0:
				query+= text
				
			# - FR-II
			text= str(n_frii) + " radio galaxies of FR-II type located at these normalized bounding box pixel coordinates (x,y,w,h): "
			coords= ''
			for i in range(n_frii):
				bbox= obj_info["FR-II"]["bboxes"][i]
				bbox_str= (str(bbox)[1:-1])
				coords+= "[" + bbox_str + "]"
				if i!=n_frii-1:
					coords+= ', '
				else:
					coords+= '. '
		
			text+= coords
			text+= "FR-II type radio sources are " + obj_info["FR-II"]["description"] + ".\n"
			if n_frii>0:
				query+= text
				
			# - FR-x
			text= str(n_frx) + " radio galaxies of FR-x type located at these normalized bounding box pixel coordinates (x,y,w,h): "
			coords= ''
			for i in range(n_frx):
				bbox= obj_info["FR-x"]["bboxes"][i]
				bbox_str= (str(bbox)[1:-1])
				coords+= "[" + bbox_str + "]"
				if i!=n_frx-1:
					coords+= ', '
				else:
					coords+= '. '
		
			text+= coords
			text+= "FR-x type radio sources are " + obj_info["FR-x"]["description"] + ".\n"
			if n_frx>0:
				query+= text
		
		# - Define additional prompt requirements
		prompt_requirements= "Include in the description only the objects given in the above list. Use an astronomical scientific style and terms like top/bottom, left/right or image width/height fractions or percentages to describe the source object positions, rather than their exact bounding box coordinates. Avoid lengthy explanations or preambles and special unicode or ascii characters. "
		
		query+= prompt_requirements
		
		# - Generate text	
		print("--> Processing image %s " % (filename))
		print(query)
		print("")
		
		if args.model_type=="llama":
			gen_description= run_llama_model_query(
				query, 
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
				query,
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
				
		gen_description= gen_description.strip('\n')
		
		print("description (LLAMA-generated)")
		print(gen_description)
		print("")
		
		q1= {"from": "human", "value": "<image>\n" + random.choice(description_list)}
		a1= {"from": "gpt", "value": gen_description}
		
		# ---------------------------------------
		# - Source bounding boxes
		# .......................................
		# - Compact
		qbbox_compact= {"from": "human", "value": "Please provide the bounding box coordinates of all compact sources present in the image."}
	
		coords= ''
		for i in range(n_compact):
			bbox= obj_info_caesar["COMPACT"]["bboxes"][i]
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
			bbox= obj_info_caesar["EXTENDED"]["bboxes"][i]
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
			bbox= obj_info_caesar["EXTENDED-MULTISLAND"]["bboxes"][i]
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
			bbox= obj_info_caesar["ARTEFACT"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_artefact-1:
				coords+= ', '
		
		response= coords
		if coords=="":
			response= "The image does not contain imaging artefacts."
		else:
			response= "The image contains imaging artefacts. Their bounding box coordinates are: " + coords
		
		abbox_artefact= {"from": "gpt", "value": response}
		
		# - Flagged 
		qbbox_flagged= {"from": "human", "value": "Does the image contain spurious sources or any imaging artefact around bright sources? Please provide their bounding box coordinates."}
	
		coords= ''
		for i in range(n_flagged):
			bbox= obj_info_caesar["FLAGGED"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_flagged-1:
				coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain radio sources to be flagged."
		else:
			response= "The image contains radio sources that are to be flagged as poorly images or significantly overlapping with imaging aretfacts. Their bounding box coordinates are: " + coords
		
		abbox_flagged= {"from": "gpt", "value": response}
		
		# - FR-I
		qbbox_fri= {"from": "human", "value": "Please provide the bounding box coordinates of all radio galaxies of FR-I type present in the image."}
	
		coords= ''
		for i in range(n_fri):
			bbox= obj_info["FR-I"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_fri-1:
				coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain FR-I type radio galaxies."
					
		abbox_fri= {"from": "gpt", "value": response}
		
		# - FR-I (extended)
		qbbox_fri_ext= {"from": "human", "value": "Can you provide the bounding box coordinates of radio galaxies of FR-I type present in the image that have an extended morphology? "}
	
		coords= ''
		for i in range(n_fri):
			smorph= obj_info["FR-I"]["smorph"][i]
			bbox= obj_info["FR-I"]["bboxes"][i]
			if smorph=="EXTENDED" or smorph=="EXTENDED-MULTISLAND":
				bbox_str= (str(bbox)[1:-1])
				coords+= "[" + bbox_str + "]"
				if i!=n_fri-1:
					coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain FR-I type radio galaxies of extended morphology."
					
		abbox_fri_ext= {"from": "gpt", "value": response}
		
		# - FR-II
		qbbox_fri= {"from": "human", "value": "Please provide the bounding box coordinates of all radio galaxies of FR-II type present in the image."}
	
		coords= ''
		for i in range(n_frii):
			bbox= obj_info["FR-II"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_frii-1:
				coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain FR-II type radio galaxies."
					
		abbox_frii= {"from": "gpt", "value": response}
		
		# - FR-II (extended)
		qbbox_frii_ext= {"from": "human", "value": "Can you provide the bounding box coordinates of radio galaxies of FR-II type present in the image that have an extended morphology? "}
	
		coords= ''
		for i in range(n_frii):
			smorph= obj_info["FR-II"]["smorph"][i]
			bbox= obj_info["FR-II"]["bboxes"][i]
			if smorph=="EXTENDED" or smorph=="EXTENDED-MULTISLAND":
				bbox_str= (str(bbox)[1:-1])
				coords+= "[" + bbox_str + "]"
				if i!=n_frii-1:
					coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain FR-II type radio galaxies of extended morphology."
					
		abbox_frii_ext= {"from": "gpt", "value": response}
		
		# - FR-x
		qbbox_fri= {"from": "human", "value": "Please provide the bounding box coordinates of all radio galaxies of FR-x type present in the image."}
	
		coords= ''
		for i in range(n_frx):
			bbox= obj_info["FR-x"]["bboxes"][i]
			bbox_str= (str(bbox)[1:-1])
			coords+= "[" + bbox_str + "]"
			if i!=n_frx-1:
				coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain FR-x type radio galaxies."
					
		abbox_frx= {"from": "gpt", "value": response}
		
		# - FR-x (extended)
		qbbox_frx_ext= {"from": "human", "value": "Can you provide the bounding box coordinates of radio galaxies of FR-x type present in the image that have an extended morphology? "}
	
		coords= ''
		for i in range(n_frx):
			smorph= obj_info["FR-x"]["smorph"][i]
			bbox= obj_info["FR-x"]["bboxes"][i]
			if smorph=="EXTENDED" or smorph=="EXTENDED-MULTISLAND":
				bbox_str= (str(bbox)[1:-1])
				coords+= "[" + bbox_str + "]"
				if i!=n_frx-1:
					coords+= ', '
			
		response= coords
		if coords=="":
			response= "The image does not contain FR-x type radio galaxies of extended morphology."
					
		abbox_frx_ext= {"from": "gpt", "value": response}
		
		#########################################
		# - Add all messages to collection
		#########################################
		conversations= [
			q1, a1,
			qbbox_compact, abbox_compact,
			qbbox_extended, abbox_extended,
			qbbox_extendedmulti, abbox_extendedmulti,
			qbbox_artefact, abbox_artefact,
			qbbox_flagged, abbox_flagged,
			qbbox_fri, abbox_fri,
			qbbox_fri_ext, abbox_fri_ext,
			qbbox_frii, abbox_frii,
			qbbox_frii_ext, abbox_frii_ext,
			qbbox_frx, abbox_frx,
			qbbox_frx_ext, abbox_frx_ext,
		]
		outdict["conversations"]= conversations
	
		# - Append to outdata
		outdata.append(outdict)

	#===========================
	#==   SAVE OUTPUT
	#===========================
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

