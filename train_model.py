import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.extract_features import extract_features  # Pastikan path benar

# === 1. Load Dataset ===
DATASET_DIR = "dataset/train"
CATEGORIES = ["mentah", "setengah-matang", "matang", "terlalu-matang"]
X, y = [], []

for category in CATEGORIES:
    category_path = os.path.join(DATASET_DIR, category)
    label = CATEGORIES.index(category)

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        features = extract_features(img)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# === 2. Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Train Model K-NN ===
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# === 4. Evaluasi Model ===
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# === 5. Simpan Model ===
os.makedirs("models", exist_ok=True)
joblib.dump(knn, "models/knn_model.pkl")
print("Model saved as models/knn_model.pkl")
