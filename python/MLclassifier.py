"""Machine learning classification of damaged and undamaged artworks.
Comparison between performances of a Support Vector Machine and a Random Forest algorithm,
these models being trained on a dataset of extracted features."""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

def plot_roc_curves(models, X, y, cv, script_dir, output_prefix):
    """Generate and save a comparison of ROC curves using cross-validated predictions.
    Args:
        models (dict): dictionary where keys are model names (str) and values are Pipeline objects
        X (pd.DataFrame): feature matrix used for evaluation
        y (pd.Series): ground truth binary labels
        cv: cross-validation generator
        script_dir (Path): Pathlib object pointing to the output directory
        output_prefix (str): prefix for the saved filename"""
    
    plt.figure(figsize=(8, 6))
    # Get current axis to plot all curves on the same figure
    ax = plt.gca()

    # Iterate over the models and plot each ROC curve
    for name, clf in models.items():
        # Get predicted probabilities for each fold
        y_probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
        
        # Use the probabilities of the positive class (column 1)
        (RocCurveDisplay.from_predictions(y, y_probs[:, 1], ax=ax)).line_.set_label(name)
    
    # Plot the diagonal (i.e. random guessing)
    plt.plot([0, 1], [0, 1], "k--", label="Chance level")

    # Plot and save the figure in specified directory
    plt.title("Cross-Validated ROC Curves Comparison", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(script_dir / f"{output_prefix}_roc_comparison.png", dpi=300)
    plt.show()

def run_classification(file_path):
    """Load features, train classification models using cross-validation, and plot results.
    Args: 
        file_path (str): path to the csv file containing features and labels
    Returns: 
        results (pd.DataFrame): a summary table of each model's performance"""

    script_dir = Path(__file__).resolve().parent

    # Load dataset
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset in {path}")
    df = pd.read_csv(path)
    output_prefix = path.stem

    # Define and separate features (X) and labels (y)
    X = df.drop('Label', axis=1)
    y = df['Label']

    models = {
        # Dictionary with model name as key and Pipeline as the related value
        "SVM": Pipeline([
            # Standardise all features to have mean = 0 and standard deviation = 1
            ('scaler', StandardScaler()), 
            # Use a Support Vector Classifier with Radial Basis Function kernel for non-linear decision boundaries
            ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))]),
        "RF": Pipeline([
            # No scaling is required for a Random Forest Classifier
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))])
    }

    # Setup cross validation maintaining class proportion across folds and define some metrics
    cval = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Compute averaged metrics for each model in the dictionary
    results = {
        name: pd.DataFrame(cross_validate(clf, X, y, cv=cval, scoring=metrics)).mean() 
        for name, clf in models.items()
    }
    
    plot_roc_curves(models, X, y, cval, script_dir, output_prefix)

    # Create a DataFrame for direct comparison, with metrics expressed as percentages
    summary = (pd.DataFrame(results).T.drop(columns=['fit_time', 'score_time']))*100
    summary = summary.round(2)
    print("MODEL COMPARISON\n", summary)
    summary.to_csv(script_dir / f"{output_prefix}_metrics.csv")

    # Compute predictions to calculate confusion matrix
    predictions = {name: cross_val_predict(clf, X, y, cv=cval) 
                   for name, clf in models.items()}

    # Setup visualisation
    fig, axes = plt.subplots(1, len(models), figsize=(12, 5))
    
    # Iterate over dictionary's keys to compute and display confusion matrixes
    for i, (name, y_pred) in enumerate(predictions.items()):
        con_mat = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=["Damaged", "Undamaged"])
        disp.plot(ax=axes[i], cmap='Blues' if i == 0 else 'Greens', colorbar=False)
        axes[i].set_title(f"Confusion Matrix: {name}")

    plt.tight_layout()
    plt.savefig(script_dir / f"{output_prefix}_confusion_matrices.png")
    plt.show()
    
    return summary

if __name__ == "__main__":
    #DATA_PATH = "C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/project_dataset/features_manual.csv"
    DATA_PATH = "C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/project_dataset/features_cnn.csv"
    run_classification(DATA_PATH)
