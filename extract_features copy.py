import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# === Fungsi Ekstraksi Fitur ===
def extract_features(image):
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ekstrak fitur GLCM
    glcm_features = extract_glcm_features(gray)

    # Ekstrak fitur HSV
    hsv_features = extract_hsv_features(image)

    # Gabungkan semua fitur dalam satu array
    return np.hstack((glcm_features, hsv_features))

# === Fungsi Ekstraksi Fitur GLCM ===
def extract_glcm_features(gray_image):
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]

# === Fungsi Ekstraksi Fitur HSV ===
def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])  # Hue
    s_mean = np.mean(hsv[:, :, 1])  # Saturation
    v_mean = np.mean(hsv[:, :, 2])  # Value
    return [h_mean, s_mean, v_mean]
