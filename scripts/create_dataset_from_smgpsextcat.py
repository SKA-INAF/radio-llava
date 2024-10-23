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

## LOGGER
logger = logging.getLogger(__name__)

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
	parser.add_argument('--generate_text_variations', dest='generate_text_variations', action='store_true', help='Generate text variations using LLAMA model (default=false)')	
	parser.set_defaults(generate_text_variations=False)
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
	if generate_text_variations:
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
	
		# - Add all messages to collection
		conversations= [
			q1, a1,
			q2, a2,
			q3, a3
		]
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
