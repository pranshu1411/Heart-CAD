import os
import scipy.io as sio
import pandas as pd
import numpy as np

def extract_anfis_metrics():
    mat_path = '../results/final_model_workspace.mat'
    if not os.path.exists(mat_path):
        print("Cannot find MATLAB workspace mat file.")
        return

    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    
    if 'final_results' not in mat:
        print("final_results not found in MAT file")
        return
        
    res = mat['final_results']
    metrics = res.metrics
    
    # metrics fields: Acc, Sens, Spec, Prec, F1, AUC
    mean_acc = np.mean(metrics.Acc)
    mean_sens = np.mean(metrics.Sens)
    mean_spec = np.mean(metrics.Spec)
    mean_prec = np.mean(metrics.Prec)
    mean_f1 = np.mean(metrics.F1)
    mean_auc = np.mean(metrics.AUC)
    
    data = [{
        "Accuracy": mean_acc,
        "Precision": mean_prec,
        "Recall": mean_sens,  # Recall = Sens
        "Sensitivity": mean_sens,
        "Specificity": mean_spec,
        "F1-Score": mean_f1,
        "ROC-AUC": mean_auc
    }]
    
    df = pd.DataFrame(data)
    csv_path = '../results/baseline_v1/FA_ANFIS_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Successfully extracted ANFIS metrics and saved to {csv_path}")

if __name__ == '__main__':
    extract_anfis_metrics()
