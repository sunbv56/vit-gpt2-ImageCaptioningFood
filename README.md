
---

# ViT-GPT2 Image Captioning for Food

This project implements an image captioning model using a Vision Transformer (ViT) for image feature extraction and GPT-2 for generating captions. The model is fine-tuned on the BBC Good Food dataset and is designed to automatically generate captions for food images. The model is publicly available on Hugging Face, and a web interface is provided to interact with the model.

You can access the model on Hugging Face here: [ViT-GPT2 Image Captioning for Food](https://huggingface.co/sunbv56/vit-gpt2-imagecaptioningfood)

Watch the demo here: [YouTube Demo](https://youtu.be/SFwY1Xytldg?si=apvIjWunsLEgUirT)

## Project Description

This project has two primary components:

1. **Fine-tuning the ViT-GPT2 Model**: The file `vit-gpt2-imagecaptioningfood.ipynb` contains the code for fine-tuning the ViT-GPT2 model on the BBC Good Food dataset. This notebook processes food images and generates captions for them using a Vision Transformer model (ViT) for feature extraction and GPT-2 for caption generation.

2. **Web Interface for Image Captioning**: The files `web/client.html` and `web/server.py` provide a simple web interface to interact with the model. The web interface allows users to upload food images, and it will return generated captions from the fine-tuned model.

## Project Structure

- **vit-gpt2-imagecaptioningfood.ipynb**: 
  - Fine-tuning the ViT-GPT2 model.
  - Preprocessing the BBC Good Food dataset and training the model.
  - Saving the fine-tuned model for use in the web interface.

- **web/client.html**: 
  - The user interface that allows interaction with the image captioning model.
  - Allows users to upload images for caption generation.

- **web/server.py**: 
  - The backend server that receives image requests from the front-end and sends them to the model for caption generation.

- **data/22k_FoodCaptionData_bbcgoodfood.csv**: 
  - A CSV file containing the BBC Good Food dataset with images and associated captions used for training the model.

- **data/load_data_api.py**: 
  - Contains functions for loading and processing the food dataset, preparing it for training.

## Setting Up the Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sunbv56/vit-gpt2-imagecaptioningfood.git
   cd vit-gpt2-imagecaptioningfood
   ```

2. **Install the required libraries**:
   After setting up a virtual environment, install the necessary libraries using the following command:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes the following libraries:
   - `sacrebleu`: For evaluating the performance of the model.
   - `transformers==4.20.1`: For using Hugging Face’s pre-trained models (ViT, GPT-2).
   - `datasets`: For loading and processing datasets.
   - `requests`: For sending HTTP requests.
   - `seaborn`: For visualizing results.
   - `numpy`, `pandas`: For handling data.
   - `Pillow`: For image processing.
   - `tqdm`: For displaying progress bars.
   - `matplotlib`: For visualizing the training process and results.
   - `scikit-learn`: For machine learning utilities.
   - `torch` and `torchvision`: For deep learning and image processing.

## Training the Model

1. **Open the `vit-gpt2-imagecaptioningfood.ipynb` file**:
   - Use Jupyter Notebook or Google Colab to open and run the notebook.
   - The notebook provides the code for fine-tuning the ViT-GPT2 model on the food image dataset.
   - The model is trained using PyTorch and the Hugging Face `transformers` library.

2. **Run the entire notebook** to train the model:
   - This includes data preprocessing, model training, and evaluation.
   - After training, the model will be saved.

## Deploying the Web Interface

1. **Run the Web Server**:
   - After fine-tuning the model, you can use the `web/server.py` to run a Flask server.
   - The server will handle requests from the front-end (`client.html`), pass the images to the model, and return the generated captions.

2. **Using the Web Interface**:
   - Open the `client.html` file in a browser.
   - Upload a food image, and the model will generate a caption for the image based on the fine-tuned model.

## Libraries Used

- **sacrebleu**: For evaluating the quality of generated captions.
- **transformers**: For fine-tuning the ViT-GPT2 model and handling pre-trained models.
- **datasets**: For loading and processing the food image dataset.
- **requests**: For handling HTTP requests between the front-end and the server.
- **seaborn**: For visualizing the results.
- **numpy**, **pandas**: For data processing.
- **Pillow**: For loading and processing images.
- **tqdm**: For displaying progress bars during training and evaluation.
- **matplotlib**: For visualizing training results.
- **scikit-learn**: For machine learning utilities.
- **torch**, **torchvision**: For model training and image preprocessing.

## Contributing

Contributions are welcome! If you find any issues, have suggestions for improvements, or want to add new features, please create an issue or submit a pull request.

---
