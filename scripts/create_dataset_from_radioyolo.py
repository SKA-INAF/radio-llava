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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input data filelist with images') 
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, default=-1, type=int, help='Max number of processed images') 
	
	# - Run options
	parser.add_argument('-model','--model', dest='model', required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help='LLAMA model used to generate variations') 
	parser.add_argument('-device_map','--device_map', dest='device_map', required=False, default="auto", type=str, help='Device map used when loading model') 
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, default=0.2, type=float, help='Temperature parameter') 
	
	parser.add_argument('-max_new_tokens','--max_new_tokens', dest='max_new_tokens', required=False, default=1024, type=int, help='The max number of tokens to be generated') 
	parser.add_argument('-top_p','--top_p', dest='top_p', required=False, default=1.0, type=float, help='If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation') 
	parser.add_argument('-top_k','--top_k', dest='top_k', required=False, default=20, type=int, help='The number of highest probability vocabulary tokens to keep for top-k-filtering') 
	parser.add_argument('-temperature','--temperature', dest='temperature', required=False, default=0.2, type=float, help='Temperature parameter') 
	parser.add_argument('-penalty','--penalty', dest='penalty', required=False, default=1.2, type=float, help='The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0 and 1.0 penalizes prompt tokens') 
	
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
	logger.info("Loading model %s ..." % (model_id))
	model, tokenizer= load_llama_model(model_id, args.device_map)

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
			"ARTEFACT": {"count": 0, "bboxes": []},
			"COMPACT": {"count": 0, "bboxes": []},
			"EXTENDED": {"count": 0, "bboxes": []},
			"EXTENDED-MULTISLAND": {"count": 0, "bboxes": []},
			"FLAGGED": {"count": 0, "bboxes": []}		
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
		query= "Consider a radio astronomical image extracted from a radio-continuum survey map produced by an interferemoter telescope. Let's consider these possible classes of radio sources that can be contained in a radio astronomical image: \n COMPACT: single-island isolated point- or slightly resolved compact radio sources, eventually hosting one or more blended components, each with morphology resembling the synthesized beam shape of the image; \n EXTENDED: radio sources with a single-island extended morphology, eventually hosting one or more blended components, with some deviating from the synthesized beam shape; \n EXTENDED-MULTISLAND: including radio sources with an extended morphology, consisting of more (point-like or extended) islands, each one eventually hosting one or more blended components; \n SPURIOUS: spurious sources, due to artefacts introduced in the radio image by the imaging process, having a ring-like or elongated compact morphology; \n FLAGGED: including single-island radio sources, with compact or extended morphology, that are poorly imaged and largely overlapping with close imaging artefacts. \n Can you generate a detailed description of a radio astronomical image containing the following list of radio source objects, each identified by a class label and normalized bounding box pixel coordinates (x,y,w,h): \n "
		
		#query= "Can you write a brief description of a radio astronomical image containing the following list of radio source objects, each represented by a class label and normalized bounding boxes (x,y,w,h)? "
		#query= "Can you write a brief text that describes a radio astronomical image containing the following list of radio source objects, each represented by a class label and normalized bounding boxes in pixel coordinates (x,y,w,h)? "
		
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
			
		query+= "Use terms like top/bottom, left/right or image width/height fractions or percentages to report source object positions, rather than their exact bounding box coordinates. Please report just the description text using an astronomical scientific style, without any prefix, preamble or explanation. "
			
		print("--> query ")
		print(query)
		
		#print("obj_info")
		#print(obj_info)
		
		gen_description= run_llama_model_query(
			query, 
			model, tokenizer, 
			temperature
			do_sample=True,
			temperature=args.temperature,
			max_new_tokens=args.max_new_tokens,
			top_p=args.top_p,
			top_k=args.top_k,
			penalty=args.penalty
		)
		
		print("description (model-generated)")
		print(gen_description)
			
		q1= {"from": "human", "value": "<image>\n" + random.choice(description_list)}
		a1= {"from": "gpt", "value": gen_description}
		###a1= {"from": "gpt", "value": ""}
		
		
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
		qbbox_flagged= {"from": "human", "value": "Does the image contain spurious sources or any imaging artefact around bright sources? Please provide their bounding box coordinates."}
	
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
			response= "The image contains radio sources that are to be flagged as poorly images or significantly overlapping with imaging aretfacts. Their bounding box coordinates are: " + coords
		
		abbox_flagged= {"from": "gpt", "value": response}
		
		
		# - Add all messages to collection
		conversations= [
			q1, a1,
			qbbox_compact, abbox_compact,
			qbbox_extended, abbox_extended,
			qbbox_extendedmulti, abbox_extendedmulti,
			qbbox_artefact, abbox_artefact,
			qbbox_flagged, abbox_flagged
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
		
