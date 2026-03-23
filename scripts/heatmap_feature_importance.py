import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

def main():
    cleveland_path = "../processed-data/cleveland_processed.csv"
    indian_path = "../processed-data/indian_processed.csv"

    if not os.path.exists(cleveland_path) or not os.path.exists(indian_path):
        print("Data files not found. Please ensure '../processed-data/' directory contains the required CSV files.")
        return

    df_cleveland = pd.read_csv(cleveland_path)
    df_indian = pd.read_csv(indian_path)

    print("Data loaded successfully.")

    X_cleveland = df_cleveland.drop('target', axis=1)
    y_cleveland = df_cleveland['target']

    X_indian = df_indian.drop('target', axis=1)
    
    scaler = StandardScaler()
    scaler.fit(X_indian) # Fitting on indian_processed for standardization
    
    X_cleveland_scaled = scaler.transform(X_cleveland)
    X_cleveland_scaled_df = pd.DataFrame(X_cleveland_scaled, columns=X_cleveland.columns)
    
    df_scaled_full = X_cleveland_scaled_df.copy()
    df_scaled_full['target'] = y_cleveland.values

    print("Standardization complete.")

    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df_scaled_full.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap\n(Cleveland Trained Data - Standardized via Indian Data)')
    plt.tight_layout()
    os.makedirs("../analysis_images/general", exist_ok=True)
    heatmap_file = '../analysis_images/general/correlation_heatmap.png'
    plt.savefig(heatmap_file, dpi=300)
    print(f"Saved correlation heatmap: {heatmap_file}")
    plt.close()

    # 2. Feature Importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_cleveland_scaled_df, y_cleveland)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [X_cleveland.columns[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    
    # Use seaborn barplot to plot importance
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette='viridis', legend=False)
    
    plt.title('Feature Importances\n(Random Forest trained on Cleveland Data)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    importance_file = '../analysis_images/general/feature_importance.png'
    plt.savefig(importance_file, dpi=300)
    print(f"Saved feature importance plot: {importance_file}")
    plt.close()

if __name__ == "__main__":
    main()
