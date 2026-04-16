import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def load_and_prepare_data():
    cleveland = pd.read_csv("../processed-data/cleveland_processed.csv")
    X = cleveland.drop("target", axis=1)
    y = cleveland["target"]
    return X, y


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=100,
                eval_metric="logloss",
                random_state=42,
            )),
        ]),
    }


def evaluate_models(X, y, models, cv_folds=10):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"Training {name} …")

        y_pred  = cross_val_predict(model, X, y, cv=skf, method="predict")
        y_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        
        # Save exact predictions for Statistical checking
        pd.DataFrame({
            "True_Label": y,
            "Pred_Label": y_pred
        }).to_csv(f"../results/baseline_v1/{name}_predictions.csv", index_label="Row_ID")

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            "Accuracy":    accuracy_score(y, y_pred),
            "Precision":   precision_score(y, y_pred, zero_division=0),
            "Recall":      recall_score(y, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1-Score":    f1_score(y, y_pred, zero_division=0),
            "ROC-AUC":     roc_auc_score(y, y_proba),
            "Confusion Matrix": cm,
            "y_proba": y_proba,
        }

    return results


def main():
    X_all, y = load_and_prepare_data()
    
    # Define feature sets
    fa_features = ["age", "sex", "thalach", "ca"]
    
    # Make sure we only grab features that exist (to avoid KeyError if the list differs)
    X_opt = X_all[fa_features] if all(f in X_all.columns for f in fa_features) else X_all
    
    feature_configs = {
        "ALL": X_all,
        "OPT": X_opt
    }
    
    models = get_models()
    
    save_dir = "../analysis_images/baseline_v1"
    csv_dir = "../results/baseline_v1"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]

    for config_name, X_current in feature_configs.items():
        print(f"\n--- Running evaluations for feature set: {config_name} ({X_current.shape[1]} features) ---")
        results = evaluate_models(X_current, y, models, cv_folds=10)

        # Save individual CSVs for each model
        for name, res in results.items():
            row = {m: res[m] for m in metrics_list}
            df_csv = pd.DataFrame([row])
            clean_name = name.replace(" ", "_").upper()
            csv_path = os.path.join(csv_dir, f"{clean_name}_{config_name}_results.csv")
            df_csv.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

        plot_confusion_matrices_custom(results, save_dir, config_name)
        plot_roc_curves_custom(results, y, save_dir, config_name)

    print("\nImages saved to ../analysis_images/baseline_v1")
    print("CSV results saved to ../results/baseline_v1/")

def plot_confusion_matrices_custom(results, save_dir, config_name):
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

    plt.suptitle(f"Confusion Matrices — Cleveland ({config_name} Features, 10-Fold CV)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrices_clev_{config_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curves_custom(results, y_true, save_dir, config_name):
    plt.figure(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_true, res["y_proba"])
        auc_val = res["ROC-AUC"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — Cleveland ({config_name} Features, 10-Fold CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_curves_clev_{config_name}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
