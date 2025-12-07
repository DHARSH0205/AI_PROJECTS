from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import joblib
import os

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
MODEL_DIR = "models"
models = {
    "logistic": joblib.load(f"{MODEL_DIR}/logistic.pkl"),
    "knn": joblib.load(f"{MODEL_DIR}/knn.pkl"),
    "nb": joblib.load(f"{MODEL_DIR}/gaussian_nb.pkl"),
}

IMG_SIZE = (64, 64)


# ====== IMAGE PREPROCESSOR ======
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).flatten().reshape(1, -1)
    return arr


# ====== PREDICT ENDPOINT ======
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_model: str = Form(...)
):
    """
    selected_model can be:
    "logistic" — Logistic Regression
    "knn"      — K-Nearest Neighbors
    "nb"       — Gaussian Naive Bayes
    """

    # read file
    img_bytes = await file.read()
    features = preprocess(img_bytes)

    # get selected model
    model = models[selected_model]

    # prediction
    pred = model.predict(features)[0]

    # confidence if supported
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        conf = float(max(proba))

    return JSONResponse({
        "prediction": pred,
      #  "confidence": conf
    })