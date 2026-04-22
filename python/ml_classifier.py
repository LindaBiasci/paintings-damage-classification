"""Machine learning classification of damaged and undamaged artworks.
Comparison between performances of a Support Vector Machine and a Random Forest algorithm,
these models being trained on a dataset of extracted features."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

def plot_roc_curves(models, X, y, cv, save_path):
    """Generate and save a comparison of ROC curves using cross-validated predictions.
    Args:
        models (dict): dictionary where keys are model names (str) and values are Pipeline objects
        X (pd.DataFrame): feature matrix used for evaluation
        y (pd.Series): ground truth binary labels
        cv: cross-validation generator
        save_path (Path): Pathlib object indicating the full output file path"""

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
    plt.savefig(save_path, dpi=300)
    plt.show()

def run_classification(file_path):
    """Load features, train classification models using cross-validation, and plot results.
    Args: 
        file_path (str): path to the csv file containing features and labels
    Returns: 
        results (pd.DataFrame): a summary table of each model's performance"""

    output_dir = Path("C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/" \
    "paintings-damage-classification/results")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    cval = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Compute averaged metrics for each model in the dictionary
    results = {
        name: pd.DataFrame(cross_validate(clf, X, y, cv=cval, scoring=metrics))
        for name, clf in models.items()
    }

    roc_path = output_dir / f"{output_prefix}_roc_comparison.png"
    plot_roc_curves(models, X, y, cval, roc_path)

    # Create a unique DataFrame for comparison, with means and deviations of percentage metrics
    all_results = pd.concat(results).drop(columns=['fit_time', 'score_time'])
    summary = all_results.groupby(level=0).agg(['mean', 'std']) * 100
    summary = summary.round(2)
    pd.set_option('display.max_columns', None)
    print(f"{output_prefix} MODEL COMPARISON\n", summary)
    summary.to_csv(output_dir / f"{output_prefix}_metrics.csv")

    # Compute predictions to calculate confusion matrix
    predictions = {name: cross_val_predict(clf, X, y, cv=cval)
                   for name, clf in models.items()}

    # Setup visualisation
    _, axes = plt.subplots(1, len(models), figsize=(12, 5))

    # Iterate over dictionary's keys to compute and display confusion matrixes
    for i, (name, y_pred) in enumerate(predictions.items()):
        con_mat = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=["Damaged", "Undamaged"])
        disp.plot(ax=axes[i], cmap='Blues' if i == 0 else 'Greens', colorbar=False)
        axes[i].set_title(f"Confusion Matrix: {name}")

    plt.tight_layout()
    plt.savefig(output_dir / f"{output_prefix}_confusion_matrices.png")
    plt.show()

# Analyse feature importance in Random Forest
    if "manual" in output_prefix:
        rf_model = models["RF"].fit(X, y)
        importances = rf_model.named_steps['rf'].feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        feat_imp.plot(kind='barh', color='limegreen')
        plt.title(f"Feature Importance (Random Forest) - {output_prefix}")
        plt.tight_layout()
        plt.show()

    return summary

if __name__ == "__main__":
    BASE_DIR = Path("C:/Users/linda/Desktop/materiali università/magistrale/Computing methods for experimental physics/" \
    "paintings-damage-classification/data/extracted_features")
    run_classification(BASE_DIR / "features_manual.csv")
    run_classification(BASE_DIR / "features_cnn.csv")
