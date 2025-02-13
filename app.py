import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify
from utils.extract_features import extract_features
from utils.utils import get_panen_status

# === Load Model ===
MODEL_PATH = "models/knn_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model belum dilatih! Jalankan train_model.py terlebih dahulu.")

knn = joblib.load(MODEL_PATH)

# === Load Kategori ===
CATEGORIES = ["mentah", "setengah-matang", "matang", "terlalu-matang"]

# === Inisialisasi Flask ===
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']

    # Validasi format file gambar
    try:
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "File bukan gambar valid"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500

    img = cv2.resize(img, (128, 128))
    features = extract_features(img)

    # Prediksi kategori dan confidence score
    probabilities = knn.predict_proba([features])[0]
    max_index = np.argmax(probabilities)
    predicted_category = CATEGORIES[max_index]
    confidence_score = probabilities[max_index]

    # Menentukan status panen
    status_panen = get_panen_status(predicted_category, confidence_score)

    return jsonify({
        "prediksi": predicted_category,
        "confidence": round(float(confidence_score), 2),
        "status_panen": status_panen
    })

if __name__ == '__main__':
    app.run(debug=True)
