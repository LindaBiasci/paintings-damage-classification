"""Feature extraction using a pre-trained Convolutional Neural Network.
This script applies transfer learning using ResNet50 trained on ImageNet
to extract deep feature representations from an image dataset.
Output: a csv file containing 2048-dimensional feature vectors and their
corresponding class labels."""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
import pandas as pd
from pathlib import Path
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import load_img, img_to_array

# Configuration
INPUT_FOLDER = Path("C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/project_dataset/processed/unpaired")
OUTPUT_CSV = Path("C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/project_dataset/features_cnn.csv")
IMAGE_SIZE = (224, 224)
LABEL_MAP = {"damaged": 0, "undamaged": 1}

def main():
    """Main execution pipeline for deep feature extraction.
    Load a pretrained ResNet50 model removing the final classification layer, 
    load and preprocess images from dataset, extract deep features using forward pass, 
    associate features with class labels, save results to csv for ML classification."""
    
    # Load pre-trained CNN, remove classification layer, get a flat vector of 2048 features
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    # Load and preprocess images to match the model input format
    image_paths = list(INPUT_FOLDER.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images.")
    images = np.array([img_to_array(load_img(p, target_size=IMAGE_SIZE)) for p in image_paths])
    x = preprocess_input(images)

    # Feature extraction
    features = model.predict(x, verbose='true')

    # Build dataset, i.e. merge feature vectors with their respective class labels
    df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(features.shape[1])])
    df["Label"] = [p.parent.name for p in image_paths]
    df["Label"] = df["Label"].map(LABEL_MAP)

    # Check for unmapped labels
    if df["Label"].isnull().any():
        print("Warning: some folders did not match the LABEL_MAP")

    # Export dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset successfully exported to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
