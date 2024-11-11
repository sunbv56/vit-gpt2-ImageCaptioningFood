from flask import Flask, request, jsonify
from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoFeatureExtractor
from huggingface_hub import login
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import base64
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Thêm dòng này để cho phép CORS

# Đăng nhập bằng token truy cập (cần thực hiện trước khi chạy server)
login("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")

# Load model và các thành phần cần thiết
tokenizer = AutoTokenizer.from_pretrained("gpt2") # for text
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224") # for image
model = VisionEncoderDecoderModel.from_pretrained("sunbv56/vit-gpt2-imagecaptioningfood") # load model

# Chuyển model sang thiết bị thích hợp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return "Image Captioning API is running"

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    data = request.get_json()
    img_data = data.get('image')

    # Chuyển đổi từ base64 thành ảnh
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")

    # Tạo caption
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return jsonify({"generated_caption": generated_caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
