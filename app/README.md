# Multi-modal Vision-Language Demo

This Streamlit app allows users to upload images and interact with a pre-trained multi-modal model by running text queries. The app leverages the BLIP (Bootstrapped Language-Image Pre-training) model to process images and generate responses to user queries.

## Features
- Upload an image (PNG, JPG, or JPEG format).
- Enter a text query to interact with the uploaded image.
- Configure model parameters, such as the temperature, using a sidebar slider.

## Requirements

- Python 3.7 or later
- Required Python libraries:
  - `streamlit`
  - `transformers`
  - `pillow`

## Installation

1. Clone the repository or save the code to a local file named `app.py`:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required libraries:

   ```bash
   pip install streamlit transformers pillow
   ```

## Running the App

1. Start the Streamlit server by running the following command:

   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in the terminal (e.g., `http://localhost:8501/`) in your web browser.

## Usage

1. Upload an image by clicking the "Upload an image" button.
2. Enter a query in the text input box (e.g., "What is in the image?" or "Describe the scene.").
3. Use the sidebar slider to configure the `temperature` parameter to adjust the model's response behavior.
4. View the model's response below the query input box.

## Customization

- To modify the app or use a different model, edit the `app.py` file. Replace the BLIP model with any Hugging Face vision-language model compatible with your needs.
- Add additional sidebar elements for more parameter configurations if required.

## Example

1. Upload an image of a landscape.
2. Enter the query: "Describe the scene."
3. Adjust the temperature slider to see how the response changes.
4. View the model's response, such as "A beautiful landscape with mountains and a lake."

## Troubleshooting

- If the app does not start, ensure all dependencies are installed and the Python version is compatible.
- For issues with specific models, consult the Hugging Face documentation for the respective model.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

