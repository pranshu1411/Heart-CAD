import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def load_and_prepare_data():
    cleveland = pd.read_csv("processed-data/cleveland_normalised.csv")
    X = cleveland.drop("target", axis=1)
    y = cleveland["target"]
    return X, y


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "XGBoost":             XGBClassifier(
                                   n_estimators=100,
                                   use_label_encoder=False,
                                   eval_metric="logloss",
                                   random_state=42,
                               ),
    }


def evaluate_models(X, y, models, cv_folds=10):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"Training {name} …")

        y_pred  = cross_val_predict(model, X, y, cv=skf, method="predict")
        y_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

        results[name] = {
            "Accuracy":  accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall":    recall_score(y, y_pred, zero_division=0),
            "F1-Score":  f1_score(y, y_pred, zero_division=0),
            "ROC-AUC":   roc_auc_score(y, y_proba),
            "Confusion Matrix": confusion_matrix(y, y_pred),
        }

    return results


def plot_metric_comparison(results, save_dir):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results.keys())

    data = {m: [results[n][m] for n in model_names] for m in metrics}
    df = pd.DataFrame(data, index=model_names)

    ax = df.plot(kind="bar", figsize=(12, 6), width=0.75, colormap="Set2", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison — Cleveland Dataset (10-Fold Stratified CV)")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metric_comparison_clev.png"), dpi=300)
    plt.close()


def plot_confusion_matrices(results, save_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=res["Confusion Matrix"],
            display_labels=["No CAD", "CAD"]
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)

    plt.suptitle("Confusion Matrices — Cleveland Dataset (10-Fold Stratified CV)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices_clev.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_table(results, save_dir):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = []
    for name, res in results.items():
        rows.append([name] + [round(res[m], 4) for m in metrics])

    df = pd.DataFrame(rows, columns=["Model"] + metrics)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    ax.set_title("Model Evaluation Results — Cleveland Dataset (10-Fold Stratified CV)",
                 fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for col_idx in range(len(df.columns)):
        cell = table[0, col_idx]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    for row_idx in range(1, len(df) + 1):
        colour = "#D9E2F3" if row_idx % 2 == 1 else "#FFFFFF"
        for col_idx in range(len(df.columns)):
            table[row_idx, col_idx].set_facecolor(colour)

    plt.savefig(os.path.join(save_dir, "model_results_clev.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    X, y = load_and_prepare_data()
    models = get_models()
    results = evaluate_models(X, y, models, cv_folds=10)

    save_dir = "analysis_images"
    os.makedirs(save_dir, exist_ok=True)

    save_summary_table(results, save_dir)
    plot_metric_comparison(results, save_dir)
    plot_confusion_matrices(results, save_dir)

    print("results saved to analysis_images/")


if __name__ == "__main__":
    main()
