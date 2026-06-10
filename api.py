"""
AgroLaafi — API d'inférence Flask
Lance avec : python3 api.py
Puis ouvre  : http://localhost:5000
"""

import io
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / "frontend"))

# ── Chargement du modèle au démarrage ─────────────────────────────────────────
print("Chargement du modèle AgroLaafi...")
MODEL = tf.keras.models.load_model(str(BASE_DIR / "models" / "best_phase2.keras"))
with open(BASE_DIR / "models" / "class_names.json", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)
print(f"Modèle prêt — {len(CLASS_NAMES)} classes")


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Lit, redimensionne et normalise une image pour MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0          # normalisation MobileNetV2 [-1, 1]
    return np.expand_dims(arr, 0)  # (1, 224, 224, 3)


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR / "frontend"), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400

    try:
        img_array = preprocess(file.read())
    except Exception as e:
        return jsonify({"error": f"Image invalide : {e}"}), 400

    preds = MODEL.predict(img_array, verbose=0)[0]
    top3  = np.argsort(preds)[-3:][::-1]

    results = [
        {"disease": CLASS_NAMES[i], "confidence": round(float(preds[i]) * 100, 1)}
        for i in top3
    ]
    return jsonify({"predictions": results})


if __name__ == "__main__":
    print("\n  AgroLaafi — http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
