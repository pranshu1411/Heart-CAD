import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    cleveland_path = "../processed-data/cleveland_processed.csv"
    indian_path = "../processed-data/indian_processed.csv"

    if not os.path.exists(cleveland_path) or not os.path.exists(indian_path):
        print("Data files not found. Please ensure '../processed-data/' directory contains the required CSV files.")
        return

    df_cleveland = pd.read_csv(cleveland_path)
    df_indian = pd.read_csv(indian_path)

    df_cleveland['Dataset'] = 'Cleveland'
    df_indian['Dataset'] = 'Indian'

    df_combined = pd.concat([df_cleveland, df_indian], ignore_index=True)

    features = [col for col in df_combined.columns if col != 'Dataset']

    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        
        sns.kdeplot(
            data=df_combined, 
            x=feature, 
            hue="Dataset", 
            fill=True, 
            common_norm=False, 
            palette="muted",
            alpha=.5, 
            linewidth=0,
            ax=ax
        )
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature Distributions: Cleveland vs Indian Datasets", fontsize=18, y=1.02)
    plt.tight_layout()
    
    os.makedirs("../analysis_images", exist_ok=True)
    output_file = "../analysis_images/distribution_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distribution analysis plot saved successfully as {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
