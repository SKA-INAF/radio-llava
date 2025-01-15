#===========================
#==   IMPORT MODULES
#===========================
# - Import standard modules
import os
import sys
import subprocess
import string
import time
import signal
import datetime
import numpy as np
import random
import math
import logging
import io

# - Import custom modules
import streamlit as st
import torch

# - Import radio-llava modules
from radio_llava.utils import load_img_as_pil_rgb
from radio_llava.inference_llava import load_llavaov_model, run_llavaov_model_query

#===========================
#==   LOAD MODEL
#===========================
# - Load the pre-trained model
@st.cache_resource
def load_model(model_id, model_name="llava_qwen"):
	""" Load model """
	
	device= 'cuda' if torch.cuda.is_available() else 'cpu'
	
	model, tokenizer, image_processor= load_llavaov_model(
		model_id,
		model_name=model_name,
		device_map=device 
		#device_map="auto"
	)
	
	return model, tokenizer, image_processor



########################
##     APP
########################
# Streamlit app UI
def main():
	st.title("radio-llava Demo")
	st.write("Upload an image and enter a query to interact with the pre-trained radio-llava model.")

	# - Model configuration
	st.sidebar.title("Model Configuration")
	model_id = st.sidebar.text_input("Enter model name or path", "lmms-lab/llava-onevision-qwen2-7b-ov")

	# Button to load the model
	#if "model_loaded" not in st.session_state:
	#	st.session_state.model_loaded = False

	# - Load the model dynamically based on the input
	if st.button("Load Model"):
		try:
			model, tokenizer, image_processor= load_model(model_id)
			st.session_state.model = model
			st.session_state.tokenizer = tokenizer
			st.session_state.image_processor = image_processor
			st.session_state.model_loaded = True
			st.sidebar.success(f"Loaded model: {model_id}")
		except Exception as e:
			st.sidebar.error(f"Failed to load model: {e}")
			return

	if "model_loaded" not in st.session_state or not st.session_state.model_loaded:
		st.write("Please load the model to proceed.")
		return     

	# - Image upload
	uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

	# - Image processing parameter configuration
	st.sidebar.title("Image Processing Parameters")
	zscale= st.sidebar.checkbox("apply zscale transform?", value=True)
	zscale_contrast= st.sidebar.slider("zscale contrast", min_value=0.1, max_value=1.0, value=0.25, step=0.1)
	
	# - Model parameter configuration
	st.sidebar.title("Model Parameters")
	do_sample= st.sidebar.checkbox("do sample?", value=False)
	temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

	# Conversation history
	if "conversation_history" not in st.session_state:
		st.session_state.conversation_history = []
		
	# Clear history button
	#if st.button("Clear History"):
	#	st.session_state.conversation_history = []
        
	if uploaded_file is not None:
		##################################
		##     LOAD IMAGE
		##################################
		# - Load the uploaded image as PIL
		image= load_img_as_pil_rgb(
			uploaded_file,
			resize=False, resize_size=224, 
			apply_zscale=zscale, contrast=zscale_contrast,
			verbose=False
		)
		if image is None:
			st.sidebar.error(f"Failed to load image!")
			return
			
		#st.image(image, caption="Uploaded Image", use_column_width=True)
		st.image(image, caption="Uploaded Image", width=300)

		##################################
		##     RUN QUERY
		##################################		
		# - Text query input form and button
		#query = st.text_input("Enter your query (e.g., 'What is in the image?' or 'Describe the scene.'):")

		if "query_to_process" not in st.session_state:
			st.session_state.query_to_process = ""
            
		st.subheader("Prompt")
		col1, col2 = st.columns([10, 1])
		with col1:
			query = st.text_input("Enter your query (e.g., 'What is in the image?' or 'Describe the image content.'):")
		with col2:
			if st.button("▶", key="send_query", help="Send query"):
				st.session_state.query_to_process = query
                
			#if st.button("", key="send_query", help="Send query", use_container_width=True, label_visibility="collapsed"):
			#	st.session_state.query_to_process = query

	
		#if query:
		if st.session_state.query_to_process:
		#if "query_to_process" in st.session_state and st.session_state.query_to_process:
			query = st.session_state.query_to_process
			
			# - Run query and get response
			response= run_llavaov_model_query(
				st.session_state.model,
				st.session_state.tokenizer,
				st.session_state.image_processor, 
				image, 
				query,
				do_sample=do_sample,
				temperature=temperature,
				conv_template="qwen_2", 
				verbose=False
			)

			# - Display the model's response
			#st.subheader("Model's Response:")
			#st.write(response)

			# Update and display conversation history
			st.session_state.conversation_history.append((query, response))
			st.session_state.query_to_process = ""  # Clear the query after processing

		##################################
		##     DISPLAY CONVERSATIONS
		##################################
		st.subheader("Conversation History")
		#for i, (q, r) in enumerate(st.session_state.conversation_history):
		#	st.write(f"**User:** {q}")
		#	st.write(f"**Assistant:** {r}")
				
		history_container = st.container()
		
		history_html = "<div style='height:300px; overflow-y:auto;'>"
		for i, (q, r) in enumerate(st.session_state.conversation_history):
			history_html += (
				f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 5px;'>"
				f"<strong>User:</strong> {q}</div>"
				f"<div style='background-color: #d4edda; padding: 10px; border-radius: 10px;'>"
				f"<strong>Assistant:</strong> {r}</div>"
			)
		history_html += "</div>"
			
		# Display the conversation history
		with history_container:
			st.markdown(history_html, unsafe_allow_html=True)
				
		# Clear history button outside of the container
		if st.button("Clear History"):
			st.session_state.conversation_history = []
			history_html = "<div style='height:300px; overflow-y:auto;'></div>"
			with history_container:
				st.markdown(history_html, unsafe_allow_html=True)
                
		# Clear history button
		#if st.button("Clear History"):
		#	st.session_state.conversation_history = []
		#	# Clear the displayed history immediately
		#	history_container.empty()
		#	with history_container:
		#		st.markdown("<div style='height:300px; overflow-y:auto;'></div>", unsafe_allow_html=True)
        

if __name__ == "__main__":
	main()
