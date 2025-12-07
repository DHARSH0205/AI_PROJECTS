# Animal Image Classifier (FastAPI + Machine Learning)

This project is a simple and efficient **Animal Image Classification System** built using **classical machine learning models** and deployed using **FastAPI**.  
The system allows users to upload an image of a wild animal through a beautiful frontend interface and get a prediction from the backend API.

### 🦁 Supported Animal Classes  
The model can classify images into the following categories:

- **Lion**
- **Tiger**
- **Cheetah**
- **Leopard**

---

## 📌 Project Overview

This project follows a complete ML workflow:

1. **Dataset Preparation**  
   Images are arranged in folders based on their class labels.

2. **Model Training**  
   Three classical ML models are trained on image pixel data:
   - **Logistic Regression**
   - **K-Nearest Neighbors (KNN)**
   - **Gaussian Naive Bayes**

3. **Model Saving**  
   The trained models are saved using `joblib` for later use.

4. **Backend API (FastAPI)**  
   A backend is created with a `/predict` endpoint that:
   - Accepts an uploaded image
   - Preprocesses and flattens it
   - Predicts the animal class
   - Returns prediction + confidence (if available)

5. **Frontend UI**  
   A responsive HTML frontend is used to:
   - Upload an image  
   - Select the model  
   - Display the prediction result  

---

## 🛠 Requirements

Install required Python packages:

```bash
pip install fastapi uvicorn numpy pillow scikit-learn joblib python-multipart
```

Train the Models (If using new dataset)
```bash
python train_models.py
```
Start FastAPI Server
```bash
python -m uvicorn main:app --reload
```
You should see:
```bash
Uvicorn running on http://127.0.0.1:8000
```
Open the Frontend --> static/index.html

Upload an animal image → choose model → classify.
