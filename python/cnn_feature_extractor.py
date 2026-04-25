"""Feature extraction using a pre-trained Convolutional Neural Network.
This script applies transfer learning using ResNet50 trained on ImageNet
to extract deep feature representations from an image dataset.
Output: a csv file containing 2048-dimensional feature vectors and their
corresponding class labels."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Required before importing Keras modules
os.environ["KERAS_BACKEND"] = "torch"
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import load_img, img_to_array

# Configuration
INPUT_FOLDER = Path(
    "C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/" \
    "project_dataset/processed/unpaired")
OUTPUT_CSV = Path(
    "C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/" \
    "paintings-damage-classification/data/extracted_features/features_cnn.csv")
IMAGE_SIZE = (224, 224)
LABEL_MAP = {"damaged": 1, "undamaged": 0}

def extract_image_data(path):
    """Load and preprocess a single image for ResNet50.
    Args:
        path (Path): Pathlib object pointing to the image file
    Returns:
        np.array: preprocessed image data (if successful)"""

    try:
        img = load_img(path, target_size=IMAGE_SIZE)
        # Convert image to array and apply ResNet50 specific preprocessing
        return preprocess_input(img_to_array(img))
    except (OSError, ValueError):
        print(f"Skipping invalid image: {path.name}")
        return None

def main():
    """Main execution pipeline for deep feature extraction.
    Load a pretrained ResNet50 model removing the final classification layer, 
    load and preprocess images from dataset, extract deep features using forward pass, 
    associate features to class labels, save results to csv for ML classification."""

    # Load pre-trained CNN, remove classification layer, get a flat vector of 2048 features
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    # Find all images in the input folder
    image_paths = list(INPUT_FOLDER.rglob("*.jpg"))

    # Load and process each image into a pixel matix, store it and filter out corrupted files
    images = []
    labels = []

    for p in image_paths:
        img = extract_image_data(p)
        if img is None:
            continue

        Label = LABEL_MAP.get(p.parent.name)
        if Label is None:
            print(f"Unknown label folder: {p.parent.name}")
            continue

        images.append(img)
        labels.append(Label)

    x = np.array(images, dtype=np.float32)
    print(f"Found {len(images)} valid images.")

    # Actual feature extraction
    features = model.predict(x, batch_size=32)
    print(f"Original features shape: {features.shape}")

    # Apply dimensionality reduction, since ResNet returns 2048 features for 289 samples
    scaled_feats = StandardScaler().fit_transform(features)
    pca = PCA(n_components=50, random_state=42)
    reduced_feats = pca.fit_transform(scaled_feats)
    print(f"Reduced features shape: {reduced_feats.shape}")

    # Build dataset, i.e. map feature vectors to their correspondent class label
    df = pd.DataFrame(reduced_feats, columns=[f"pca_{i}" for i in range(reduced_feats.shape[1])])
    df["Label"] = labels

    # Export dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset successfully exported to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
