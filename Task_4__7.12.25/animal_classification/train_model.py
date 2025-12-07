import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# ==== CONFIG ====
DATA_DIR = r"C:\Users\spdre\Documents\CODINGS\AI_T&P\Day_6\animal_class\animals_dataset\train"
IMG_SIZE = (64, 64)
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==== LOAD DATASET ====
def load_dataset(folder):
    X, y = [], []
    
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        if not os.path.isdir(class_folder):
            continue

        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                features = np.array(img).flatten()     # flatten 64x64x3 → 12288
                X.append(features)
                y.append(label)
            except:
                print("Skipped:", img_path)

    return np.array(X), np.array(y)


print("Loading training data...")
X_train, y_train = load_dataset(DATA_DIR)
print("Training samples:", len(X_train))


# ==== MODELS TO TRAIN ====
models = {
    "logistic": LogisticRegression(max_iter=1000, multi_class="multinomial"),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "gaussian_nb": GaussianNB(),
}

# ==== TRAIN AND SAVE ====
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
    print(f"Saved → {MODEL_DIR}/{name}.pkl")

print("\nTraining complete! All models saved.")
