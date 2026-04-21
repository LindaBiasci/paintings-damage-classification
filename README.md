# paintings-damage-classification
This project aims to classify artworks as damaged or undamaged, comparing two approaches:
* Traditional machine learning with handcrafted features
* Deep learning feature extraction using a pre-trained CNN

## Dataset
**Original dataset:**
[Kaggle Dataset](https://www.kaggle.com/datasets/pes1ug22am047/damaged-and-undamaged-artworks)  
**Clean version:**
[Google Drive Link](https://drive.google.com/drive/folders/1MHRfAiGpdYqKGSp35vG3KYK0MP-80ScX)  
(For more information, see `dataset.txt` in the `data/` folder)

## Methodology

### Traditional pipeline (MATLAB)
* Manual feature extraction: texture, edge, sharpness, spectral and colour features
* Feature selection via statistical analysis on paired images

### Deep learning pipeline (Python)
* Automatic feature extraction using ResNet50, pre-trained on ImageNet
* Dimensionality reduction via PCA on unpaired images

### Machine learning classification (Python)
* Classification models: Support Vector Machine and Random Forest
* k-fold cross-validation on features extracted from unpaired images

### Evaluation and comparison:
* Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
* Outputs: ROC curves, Confusion matrices, metrics histogram for comparison

## How to run

#### 1. Preprocessing and manual feature extraction
_Optional_: run `jpg_converter.m` (_ignore it for_ [clean dataset](https://drive.google.com/drive/folders/1MHRfAiGpdYqKGSp35vG3KYK0MP-80ScX))  
Run `manual_feature_extractor.m` to generate `features_manual.csv`
#### 2. DNN feature extraction
Run `cnn_feature_extractor.py` to generate `features_cnn.csv`
#### 3. ML classification
Run `ml_classifier.py` to generate plots and metrics (see `results/` directory)
#### 4. Results analysis
Run `pipelines_comparison.py` to generate a comparison plot (see `results/` directory)

## Requirements
* **MATLAB** R2025b+ (Image Processing Toolbox, Statistics and Machine Learning Toolbox)
* **Python** 3.11+ (see `requirements.txt` for library dependencies)

## Author
**Linda Biasci**  
This is a _Computing Methods for Experimental Physics and Data Analysis_ project
