import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

def analyze_dataset(file_path, dataset_name):
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Analyzing {dataset_name} data...")

    X = df.drop('target', axis=1)
    y = df['target']

    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap\n({dataset_name} Data)')
    plt.tight_layout()
    out_dir = "../analysis_images/baseline_v1"
    os.makedirs(out_dir, exist_ok=True)
    heatmap_file = f'{out_dir}/correlation_heatmap_{dataset_name.lower()}.png'
    plt.savefig(heatmap_file, dpi=300)
    print(f" - Saved correlation heatmap: {heatmap_file}")
    plt.close()

    # 2. Feature Importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [X.columns[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette='viridis', legend=False)
    
    plt.title(f'Feature Importances\n(Random Forest trained on {dataset_name} Data)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    importance_file = f'{out_dir}/feature_importance_{dataset_name.lower()}.png'
    plt.savefig(importance_file, dpi=300)
    print(f" - Saved feature importance plot: {importance_file}")
    plt.close()

def main():
    datasets = {
        "Cleveland": "../processed-data/cleveland_processed.csv",
        "Indian": "../processed-data/indian_processed.csv",
        "Statlog": "../processed-data/statlog_processed.csv"
    }

    for name, path in datasets.items():
        analyze_dataset(path, name)

if __name__ == "__main__":
    main()
