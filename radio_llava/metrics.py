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

#################
##   METRICS
#################
# - See https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn
def hamming_score(y_true, y_pred):
	""" Compute the hamming score """
	return ( (y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1) ).mean()
	
def hamming_score_v2(y_true, y_pred, normalize=True, sample_weight=None):
	""" Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case (http://stackoverflow.com/q/32239577/395857)"""
	acc_list = []
	for i in range(y_true.shape[0]):
		set_true = set( np.where(y_true[i])[0] )
		set_pred = set( np.where(y_pred[i])[0] )
		tmp_a = None
		if len(set_true) == 0 and len(set_pred) == 0:
			tmp_a = 1
		else:
			num= len(set_true.intersection(set_pred))
			denom= float( len(set_true.union(set_pred)) )
			tmp_a = num/denom
			
		acc_list.append(tmp_a)
		   
	return np.nanmean(acc_list)

def multiclass_singlelabel_metrics(y_true, y_pred, target_names, labels=None):
	""" Helper function to compute single-label multi-class metrics """

	class_report= classification_report(y_true, y_pred, target_names=target_names, labels=labels, output_dict=True)
	accuracy = accuracy_score(y_true, y_pred, normalize=True) # NB: This computes subset accuracy (the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true)
	precision= precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
	recall= recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
	f1score= f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
	f1score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
	
	#roc_auc_ovr = roc_auc_score(y_true, y_pred, average = 'micro', multi_class='ovr')
	#roc_auc_ovo = roc_auc_score(y_true, y_pred, average = 'micro', multi_class='ovo')
	
	# - Return as dictionary
	metrics = {
		'class_names': target_names,
		'accuracy': accuracy,
		'recall': recall,
		'precision': precision,
		'f1score': f1score,
		'f1score_micro': f1score_micro,
		#'roc_auc_ovr': roc_auc_ovr,
		#'roc_auc_ovo': roc_auc_ovo,
		'class_report': class_report,
	}
	
	print("metrics")
	print(metrics)
	  
	return metrics
	

def multiclass_multilabel_metrics(y_true, y_pred, target_names):
	""" Helper function to compute multi-class multi-label metrics """
	
	# - Compute class report
	class_report= classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
	accuracy = accuracy_score(y_true, y_pred, normalize=True) # NB: This computes subset accuracy (the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true)
	precision= precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
	recall= recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
	f1score= f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
	f1score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
	roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
	h_loss= hamming_loss(y_true, y_pred)
	h_score= hamming_score_v2(y_true, y_pred)
	class_accuracy= [accuracy_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1]) ]  ## LIKELY NOT CORRECT!!!
	  
	# - Return as dictionary
	metrics = {
		'class_names': target_names,
		'accuracy': accuracy,
		'recall': recall,
		'precision': precision,
		'f1score': f1score,
		'f1score_micro': f1score_micro,
		'roc_auc': roc_auc,
		'h_loss': h_loss,
		'h_score': h_score,
		'accuracy_class': class_accuracy,
		'class_report': class_report,
	}
	
	print("metrics")
	print(metrics)
	  
	return metrics
	
def print_metrics(metrics):
	""" Print of dict/list metrics """
		
	# - Find class names
	class_names= []
	for key in metrics:
		if 'class_names' in key:
			class_names= metrics[key]		
		
	max_class_name_length= max([len(x) for x in class_names], default=0)
	print("max_class_name_length=%d" % (max_class_name_length))
		
	metric_names_for_format= []
	metric_values_for_format= []
	for x in metrics.keys():
		if "class_names" in x:
			continue
		elif "accuracy_class" in x:
			continue
		elif "class_report" in x:
			continue
		else:
			metric_names_for_format.append(x)
			metric_values_for_format.append(metrics[x])
			
	k_width = max(len(str(x)) for x in metric_names_for_format) + max_class_name_length
	v_width = max(len(str(x)) for x in metric_values_for_format)
			
	print("log_metrics-->class_names")
	print(class_names)
		
	# - Print formatted metrics
	for key in sorted(metrics.keys()):
		if "class_names" in key:
			continue
		elif "class_report" in key:
			# - Print class #instances
			for class_name in class_names:
				ninstances= metrics[key][class_name]['support']
				metric_name= key.split("_")[0] + '_nsamples_class_' + class_name
				metric_val= ninstances
				print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
			# - Print class precision
			for class_name in class_names:
				precision= metrics[key][class_name]['precision']
				metric_name= key.split("_")[0] + '_precision_class_' + class_name
				metric_val= precision
				print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
			# - Print class recall
			for class_name in class_names:
				recall= metrics[key][class_name]['recall']
				metric_name= key.split("_")[0] + '_recall_class_' + class_name
				metric_val= recall
				print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
			# - Print class F1score
			for class_name in class_names:
				f1score= metrics[key][class_name]['f1-score']
				metric_name= key.split("_")[0] + '_f1score_class_' + class_name
				metric_val= f1score
				print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
			
		elif "confusion_matrix" in key:
			print(f"  {metric_name: <{k_width}} ")
			print(metrics[key])
			
		elif "confusion_matrix_norm" in key:
			print(f"  {metric_name: <{k_width}} ")
			print(metrics[key])
				
		else:
			metrics_str= str(metrics[key])
			print(f"  {key: <{k_width}} = {metrics_str:>{v_width}}")

