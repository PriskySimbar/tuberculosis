from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os
import requests

app = Flask(__name__)

# ===============================
# 🔹 Setup model path & Drive URL
# ===============================
MODEL_PATH = "tb_model.h5"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1kzkhcAiiHoddmzRSSu0HLTsx2XO9jrAA"

# ===============================
# 🔹 Download model if not exists
# ===============================
def download_model():
    print("Downloading model from Google Drive...")
    r = requests.get(DRIVE_URL, allow_redirects=True)
    r.raise_for_status()  # Raise error kalau gagal
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Download completed.")

if not os.path.exists(MODEL_PATH):
    download_model()

# ===============================
# 🔹 Load model
# ===============================
model = load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

# ===============================
# 🔹 Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    message = None

    if request.method == "POST":
        if "file" not in request.files:
            message = "❌ No file uploaded"
        else:
            file = request.files["file"]
            if file.filename == "":
                message = "❌ Empty filename"
            else:
                try:
                    # Baca gambar
                    img = Image.open(io.BytesIO(file.read()))
                    img = img.resize(IMG_SIZE)
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    prediction = model.predict(img_array)
                    prob = prediction[0][0]  # output sigmoid

                    if prob > 0.5:
                        message = f"✅ TB Detected with probability {prob:.2f}"
                    else:
                        message = f"🫁 Normal (No TB) with probability {1 - prob:.2f}"
                except Exception as e:
                    message = f"⚠️ Error: {str(e)}"

    return render_template("index.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)
