import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import os

def run_mcnemar_test():
    reports_dir = "../results/baseline_v1"
    anfis_pred_file = "../results/anfis_predictions.csv"
    
    if not os.path.exists(anfis_pred_file):
        print("ANFIS predictions file not found. You must run run_fa_opt_anfis.m first to generate it.")
        return

    anfis_df = pd.read_csv(anfis_pred_file).set_index("Row_ID")
    
    baseline_files = [f for f in os.listdir(reports_dir) if f.endswith("_predictions.csv")]
    
    if not baseline_files:
        print(f"No baseline prediction files found in {reports_dir}.")
        return

    print("=== McNEMAR'S TEST: FA-OPT-SC-ANFIS vs BASELINES ===")
    print("Null Hypothesis: Both models have the same error rate.")
    print("If p-value < 0.05, reject Null Hypothesis (Significant difference).")
    print("-" * 65)
    
    print(f"{'Baseline Model':<30} | {'Statistic':<10} | {'p-value':<10} | {'Significant?':<12}")
    print("-" * 65)

    for b_file in baseline_files:
        model_name = b_file.replace("_predictions.csv", "")
        b_df = pd.read_csv(os.path.join(reports_dir, b_file)).set_index("Row_ID")
        
        # Merge on Row_ID to ensure we are comparing the exact same instances
        merged = anfis_df.join(b_df, lsuffix="_anfis", rsuffix="_base", how="inner")
        
        # Drop NaNs just in case
        merged = merged.dropna(subset=["Pred_Label_anfis", "Pred_Label_base", "True_Label_anfis"])
        
        y_true = merged["True_Label_anfis"]
        y_anfis = merged["Pred_Label_anfis"]
        y_base = merged["Pred_Label_base"]
        
        # Contingency table
        #          Base_Right | Base_Wrong
        # Anfis_R |     a     |     b
        # Anfis_W |     c     |     d
        
        anfis_correct = (y_anfis == y_true)
        base_correct = (y_base == y_true)
        
        a = sum(anfis_correct & base_correct)
        b = sum(anfis_correct & ~base_correct)
        c = sum(~anfis_correct & base_correct)
        d = sum(~anfis_correct & ~base_correct)
        
        contingency_table = [[a, b], [c, d]]
        
        # Exact McNemar's test (binomial dist) is standard
        result = mcnemar(contingency_table, exact=True)
        
        sig = "YES" if result.pvalue < 0.05 else "No"
        print(f"{model_name:<30} | {result.statistic:<10.2f} | {result.pvalue:<10.4e} | {sig:<12}")

if __name__ == "__main__":
    run_mcnemar_test()
