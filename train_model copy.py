import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from extract_features import extract_features  # Import fungsi ekstraksi fitur

# === 1. Load Dataset ===
DATASET_DIR = "dataset/train"  # Sesuaikan dengan lokasi dataset
CATEGORIES = ["matang", "mentah", "setengah-matang", "terlalu-matang"]  # Kategori kematangan

X = []
y = []

# Looping semua kategori dalam dataset
for category in CATEGORIES:
    category_path = os.path.join(DATASET_DIR, category)
    label = CATEGORIES.index(category)  # Label kategori berdasarkan indeks

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)  # Baca gambar
        if img is None:
            print(f"❌ Gagal membaca gambar: {img_path}")
            continue
        
        img = cv2.resize(img, (128, 128))  # Resize agar konsisten
        features = extract_features(img)  # Ekstrak fitur GLCM & HSV
        X.append(features)
        y.append(label)

# Konversi ke array numpy
X = np.array(X)
y = np.array(y)

print(f"✅ Dataset Loaded: {X.shape}, Labels: {y.shape}")

# === 2. Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training set: {X_train.shape}, Testing set: {X_test.shape}")

# === 3. Train Model K-NN ===
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# === 4. Evaluasi Model ===
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2f}")

# === 5. Simpan Model ===
joblib.dump(knn, "knn_model.pkl")
print("✅ Model saved as knn_model.pkl")
