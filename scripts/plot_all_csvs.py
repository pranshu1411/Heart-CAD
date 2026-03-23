import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_all_csvs():
    results_dir = '../results'
    save_dir = '../analysis_images/comparison'
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{results_dir}'.")
        return
        
    # Standardize metric names to these target keys
    target_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'ROC-AUC']
    
    model_stats = {}
    
    print(f"Found {len(csv_files)} CSV files. Parsing results...")
    
    for file in csv_files:
        model_name = os.path.basename(file).replace('_results.csv', '').replace('.csv', '').upper()
        df = pd.read_csv(file)
        
        # If there's an aggregated mean row (e.g. Fold == 0, 'Mean', 'Average'), drop it
        # so we can calculate a clean mean and std purely from the k-fold rows.
        if 'Fold' in df.columns:
            df['Fold_str'] = df['Fold'].astype(str)
            df = df[~df['Fold_str'].isin(['0', '0.0', 'Mean', 'mean', 'Average', 'average'])]
        
        # Map exact columns to standard metric names
        col_map = {
            'Accuracy': 'Accuracy',
            'Sensitivity': 'Sensitivity', 'Recall': 'Sensitivity', # treat Recall as Sensitivity
            'Specificity': 'Specificity',
            'Precision': 'Precision',
            'F1_Score': 'F1-Score', 'F1-Score': 'F1-Score', 'F1': 'F1-Score',
            'AUC': 'ROC-AUC', 'ROC-AUC': 'ROC-AUC', 'ROC_AUC': 'ROC-AUC'
        }
        
        means = []
        stds = []
        
        for tm in target_metrics:
            matched_col = None
            for col in df.columns:
                if col in col_map and col_map[col] == tm:
                    matched_col = col
                    break
            
            if matched_col and len(df) > 0:
                vals = pd.to_numeric(df[matched_col], errors='coerce').dropna()
                mean_val = vals.mean()
                std_val = vals.std() if len(vals) > 1 else 0.0
            else:
                mean_val = 0.0
                std_val = 0.0
                
            means.append(mean_val)
            stds.append(std_val)
            
        model_stats[model_name] = {
            'means': means,
            'stds': stds
        }
        print(f" - Parsed stats for: {model_name}")
    
    # Plotting Grouped Bar Chart
    n_models = len(model_stats)
    n_metrics = len(target_metrics)
    
    if n_models == 0:
        print("No valid data found to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bar_width = 0.8 / n_models
    index = np.arange(n_metrics)
    
    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (model_name, stats) in enumerate(model_stats.items()):
        pos = index + i * bar_width - (0.8 / 2) + (bar_width / 2)
        ax.bar(pos, stats['means'], bar_width, yerr=stats['stds'], 
               label=model_name, capsize=4, edgecolor='black', color=colors[i])
        
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(target_metrics, fontsize=11)
    
    # Put legend outside the plot
    ax.legend(title='Models', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'all_models_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to: {plot_path}")

    # Summary Table
    rows = []
    for model_name, stats in model_stats.items():
        row = [model_name] + [round(m, 4) for m in stats['means']]
        rows.append(row)

    table_df = pd.DataFrame(rows, columns=['Model'] + target_metrics)

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(rows) + 1.5))
    ax.axis('off')
    ax.set_title('All Models — Metric Summary Table (10-Fold Stratified CV)',
                 fontsize=14, fontweight='bold', pad=14)

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for col_idx in range(len(table_df.columns)):
        cell = table[0, col_idx]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')

    for row_idx in range(1, len(table_df) + 1):
        colour = '#D9E2F3' if row_idx % 2 == 1 else '#FFFFFF'
        for col_idx in range(len(table_df.columns)):
            table[row_idx, col_idx].set_facecolor(colour)

    table_path = os.path.join(save_dir, 'all_models_summary_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Summary table saved to: {table_path}")

if __name__ == '__main__':
    plot_all_csvs()
