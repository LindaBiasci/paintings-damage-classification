"""Feature extraction using a pre-trained Convolutional Neural Network.
This script applies transfer learning using ResNet50 trained on ImageNet
to extract deep feature representations from an image dataset.
Output: a csv file containing 2048-dimensional feature vectors and their
corresponding class labels."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
# Required before importing Keras modules
os.environ["KERAS_BACKEND"] = "torch"
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import load_img, img_to_array

# Configuration
INPUT_FOLDER = Path(
    "C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/project_dataset/processed/unpaired")
OUTPUT_CSV = Path(
    "C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/project_dataset/features_cnn.csv")
IMAGE_SIZE = (224, 224)
LABEL_MAP = {"damaged": 0, "undamaged": 1}

def extract_image_data(path):
    """Load and preprocess a single image for ResNet50.
    Args:
        path (Path): Pathlib object pointing to the image file
    Returns:
        np.array: preprocessed image data (if successful)
    """

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
    print(f"Found {len(image_paths)} images.")
    # Load and process each image into a pixel matix, store it and filter out corrupted files
    x = np.array([data for p in image_paths if (data := extract_image_data(p)) is not None])

    # Actual feature extraction
    features = model.predict(x)

    # Build dataset, i.e. map feature vectors to their correspondent class label
    df = pd.DataFrame(features).add_prefix("feat_")
    df["Label"] = [LABEL_MAP[p.parent.name] for p in image_paths]

    # Check for unmapped labels
    if df["Label"].isnull().any():
        print("Warning: some folders did not match the LABEL_MAP")

    # Export dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset successfully exported to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
