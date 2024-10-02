#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import numpy as np
import json
import logging
import logging.config
import argparse
import random
import warnings

# - SKIMAGE/SKLEARN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss

## LOGGER
logger = logging.getLogger(__name__)

##################################################
###         INFERENCE UTILS
##################################################
def process_model_output(model_output, label, label2id, classification_mode, label_modifier_fcn):
	""" Process model output """
	
	res= None
	if classification_mode=="multiclass_multilabel":
		res= process_model_output_multiclass_multilabel(model_output, label, label2id, label_modifier_fcn)
	elif classification_mode=="multiclass_singlelabel":
		res= process_model_output_multiclass_singlelabel(model_output, label, label2id, label_modifier_fcn)
	else:
		logger.error("Invalid/unknown classification mode specified!" % (classification_mode))
		res= None
	
	return res


def process_model_output_multiclass_singlelabel(model_output, label, label2id, label_modifier_fcn=None):
	""" Process model output for multi-class single-label classification """
	
	# - Process ground truth label
	if label_modifier_fcn is not None:
		label= label_modifier_fcn(label)
	
	# - Extract predicted label
	label_pred= model_output.strip("\n").strip().upper()
	
	# - Remove list of characters from the label
	char_to_be_removed = [',', '!', '.', ';']
	label_pred= label_pred.translate({ord(x): '' for x in char_to_be_removed})

	# - Check if label is correct
	if label_pred not in label2id:
		logger.warn("Unexpected label (%s) returned!" % (label_pred))
		return None
	
	# - Extract class ids
	classid= label2id[label]
	classid_pred= label2id[label_pred]
	
	return classid, classid_pred, label, label_pred
		

def filter_smorph_label(labels):
	""" Modifies the input labels for smorph dataset """

	# - First add EXTENDED label if RADIO-GALAXY is present
	labels_sel= labels.copy()
	if "RADIO-GALAXY" in labels_sel and "EXTENDED" not in labels_sel:
		labels_sel.append("EXTENDED")
	
	# - Remove undesired labels
	labels_to_be_removed= ["RADIO-GALAXY","DUBIOUS","WTF","FILAMENT","RING","ARC","ARTEFACT","BORDER","MOSAICING"]
	for item in labels_to_be_removed:
		if item in labels_sel:
			labels_sel.remove(item)
	
	# - Set label to NONE if only COMPACT/BACKGROUND labels are given
	nlabels= len(labels_sel)
	if nlabels==1 and (labels_sel[0]=="COMPACT" or labels_sel[0]=="BACKGROUND"):
		labels_sel[0]= "NONE"
	
	# - Remove undesired labels
	labels_to_be_removed= ["COMPACT","BACKGROUND"]
	for item in labels_to_be_removed:
		if item in labels_sel:
			labels_sel.remove(item)
			
	# - Check if no labels left
	if not labels_sel:
		logger.error("No labels left from original labels (%s)" % (str(labels)))
	
	return labels_sel
	

def process_model_output_multiclass_multilabel(model_output, labels, label2id, label_modifier_fcn=None):
	""" Process model output for multi-class multi-label classification """

	# - Set number of class and multilabel binarizer
	nclasses= len(label2id)
	class_names= list(label2id.keys())
	mlb = MultiLabelBinarizer(classes=np.arange(0, nclasses))

	# - Process ground truth label (label is in this case a string container labels separated by commas)
	#labels= [str(x.strip()) for x in label.split(',')]
	#if label_modifier_fcn is not None:
	#	labels= label_modifier_fcn(labels)
	
	# - Process ground truth label (label is in this case a list with labels)
	if label_modifier_fcn is not None:
		labels= label_modifier_fcn(labels)
		
	# - Process predicted label
	label_pred= model_output.strip("\n").strip().upper()
	
	# - Remove list of characters from the label
	char_to_be_removed = ['!', '.', ';']
	label_pred= label_pred.translate({ord(x): '' for x in char_to_be_removed})
	
	# - Split string into label list
	labels_pred= [str(x.strip()) for x in label_pred.split(',')]

	# - Remove duplicated classes from list
	labels_pred = list(dict.fromkeys(labels_pred))

	print("--> labels (TRUE)")	
	print(labels)
	
	print("--> labels (PRED)")
	print(labels_pred)

	# - Check if labels are correct
	if not labels_pred:
		logger.warning("Empty labels_pred, returning None!")
		return None
	
	for item in labels_pred:
		if item not in label2id:
			logger.warning("Unexpected label (%s) returned, return None!" % (item))
			return None
	
	# - Compute class ids
	class_ids= [label2id[item] for item in labels]
	class_ids.sort()
	
	class_ids_pred= [label2id[item] for item in labels_pred]
	class_ids_pred.sort()
	
	print("class_ids (TRUE)")
	print(class_ids)
	
	print("class_ids (PRED)")
	print(class_ids_pred)
	
	# - Convert ids to hot-encoding
	class_ids_hotenc= mlb.fit_transform([class_ids]) # this returns 2d numpy array
	class_ids_pred_hotenc= mlb.fit_transform([class_ids_pred])
	
	class_ids_hotenc= list(class_ids_hotenc[0])
	class_ids_pred_hotenc= list(class_ids_pred_hotenc[0])
	
	print("class_ids_hotenc (TRUE)")
	print(class_ids_hotenc)
	
	print("class_ids_hotenc (PRED)")
	print(class_ids_pred_hotenc)
	
	return class_ids_hotenc, class_ids_pred_hotenc, labels, labels_pred
	
	
	
def format_context_model_response(response, classification_mode, modifier_fcn):
	""" Format model response """
	
	res= None
	if classification_mode=="multiclass_multilabel":
		res= format_context_model_response_multiclass_multilabel(response, modifier_fcn)
	elif classification_mode=="multiclass_singlelabel":
		res= format_context_model_response_singlelabel(response, modifier_fcn)
	else:
		logger.error("Invalid/unknown classification mode specified!" % (classification_mode))
		res= None
	
	return res
	
def format_context_model_response_multiclass_singlelabel(response, modifier_fcn=None):
	""" Process model output for multi-class single-label classification """
	
	# - Response (read from json file) is the ground truth classification label
	#   Return label unless there is a modifier_fcn specified
	if modifier_fcn is None:
		return response
	else:
		return modifier_fcn(response)
	
def format_context_model_response_multiclass_multilabel(response, modifier_fcn=None):
	""" Process model output for multi-class multi-label classification """
	
	# - Response (read from json file) is a list of ground truth classification labels
	#   Apply modifier_fcn and then convert to comma-separated strings
	labels= response
	if modifier_fcn is not None:
		labels= modifier_fcn(labels)
	
	labels_str= ','.join(labels)
	
	return labels_str

