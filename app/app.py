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
	model, tokenizer, image_processor= load_llavaov_model(
		model_id,
		model_name=model_name, 
		device_map="auto"
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

	# - Load the model dynamically based on the input
	try:
		model, tokenizer, image_processor= load_model(model_id)
		st.sidebar.success(f"Loaded model: {model_id}")
	except Exception as e:
		st.sidebar.error(f"Failed to load model: {e}")
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

	if uploaded_file is not None:
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
			
		st.image(image, caption="Uploaded Image", use_column_width=True)

		# - Text query input
		query = st.text_input("Enter your query (e.g., 'What is in the image?' or 'Describe the scene.'):")

		if query:
			# - Run query and get response
			response= run_llavaov_model_query(
				model,
				tokenizer,
				image_processor, 
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

			st.subheader("Conversation History")
			for i, (q, r) in enumerate(st.session_state.conversation_history):
				st.write(f"**User:** {q}")
				st.write(f"**Assistant:** {r}")

if __name__ == "__main__":
	main()
