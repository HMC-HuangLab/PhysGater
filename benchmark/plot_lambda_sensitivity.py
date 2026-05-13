# -*- coding: utf-8 -*-
import pandas as pd


import argparse

def _parse_set_overrides():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override config/global values from CLI, e.g. --set csv_path=/data/train.csv"
    )
    args, _ = parser.parse_known_args()
    out = {}
    for item in args.set:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def _apply_overrides_to_class(cls, overrides):
    for k, v in overrides.items():
        if hasattr(cls, k):
            cur = getattr(cls, k)
            try:
                if isinstance(cur, bool):
                    val = v.lower() in {"1", "true", "yes", "y", "on"}
                elif isinstance(cur, int):
                    val = int(v)
                elif isinstance(cur, float):
                    val = float(v)
                else:
                    val = v
            except Exception:
                val = v
            setattr(cls, k, val)

def _apply_overrides_to_globals(ns, overrides):
    for k, v in overrides.items():
        if k in ns:
            cur = ns[k]
            try:
                if isinstance(cur, bool):
                    val = v.lower() in {"1", "true", "yes", "y", "on"}
                elif isinstance(cur, int):
                    val = int(v)
                elif isinstance(cur, float):
                    val = float(v)
                else:
                    val = v
            except Exception:
                val = v
            ns[k] = val
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score
import os

# English comment: see script logic.
# English comment: see script logic.
INPUT_CSV = "/mnt/data/fpdetec_V2/oof_predictions.csv" 
OUTPUT_DIR = "/mnt/data/fpdetec_V2/bench_results/lambda_sensitivity"

# English comment: see script logic.
BEST_THRESHOLD = 0.46  
SELECTED_LAMBDA = 0.14

# English comment: see script logic.
LAMBDAS = np.linspace(0.0, 1.0, 101) 
# =======================================

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
sns.set_theme(style="whitegrid", font_scale=1.2)

def calc_metrics(y_true, prob_a, prob_b, lam, threshold):
    # English comment: see script logic.
    # Score = A^lam * B^(1-lam)
    fused_score = np.exp(lam * np.log(prob_a + 1e-6) + (1 - lam) * np.log(prob_b + 1e-6))
    
    # English comment: see script logic.
    y_pred = (fused_score > threshold).astype(int)
    
    # English comment: see script logic.
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # English comment: see script logic.
    # English comment: see script logic.
    score = f1
    
    return score, rec, prec

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return

    df = pd.read_csv(INPUT_CSV)
    
    y_true = df['Ground_Truth'].values
    prob_a = df['Prob_A'].values
    prob_b = df['Prob_B'].values
    
    results = []
    print(f"Scanning Lambda (Fixed Threshold={BEST_THRESHOLD})...")
    
    for lam in LAMBDAS:
        score, rec, prec = calc_metrics(y_true, prob_a, prob_b, lam, BEST_THRESHOLD)
        results.append({
            "Lambda": lam, 
            "Score": score,  # English comment removed for consistency.
            "Recall": rec,
            "Precision": prec
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.astype(float)
    
    # English comment: see script logic.
    best_row = res_df.loc[res_df['Score'].idxmax()]
    peak_lam = best_row['Lambda']
    peak_score = best_row['Score']
    
    print(f"Peak Lambda in Plot: {peak_lam:.2f} (F1 Score: {peak_score:.4f})")
    
    # English comment: see script logic.
    plt.figure(figsize=(10, 6))
    
    # English comment: see script logic.
    sns.lineplot(data=res_df, x="Lambda", y="Score", linewidth=3, color="#d62728", errorbar=None, label="F1 Score")
    
    # English comment: see script logic.
    sns.lineplot(data=res_df, x="Lambda", y="Recall", linewidth=1.5, color="blue", alpha=0.3, linestyle="--", errorbar=None, label="Recall")
    sns.lineplot(data=res_df, x="Lambda", y="Precision", linewidth=1.5, color="green", alpha=0.3, linestyle="--", errorbar=None, label="Precision")
    
    # English comment: see script logic.
    # English comment: see script logic.
    my_score, my_rec, my_prec = calc_metrics(y_true, prob_a, prob_b, SELECTED_LAMBDA, BEST_THRESHOLD)
    
    plt.scatter(SELECTED_LAMBDA, my_score, color='blue', marker='*', s=200, zorder=10, 
                label=f"Selected $\lambda$={SELECTED_LAMBDA}\n(F1={my_score:.4f})")

    # English comment: see script logic.
    # English comment: see script logic.
    valid_indices = res_df[res_df['Recall'] >= 0.8]
    if not valid_indices.empty:
        min_valid_lam = valid_indices['Lambda'].min()
        max_valid_lam = valid_indices['Lambda'].max()
        plt.axvspan(min_valid_lam, max_valid_lam, color='gray', alpha=0.1, label="Recall $\geq$ 0.8 Region")

    # English comment: see script logic.
    plt.title(f"Parameter Sensitivity: Fusion Weight ($\lambda$)\n(Metric: F1 Score, Fixed Threshold = {BEST_THRESHOLD})", fontsize=14)
    plt.xlabel(r"Weight of Path A ($\lambda$)" + "\n" + r"$\leftarrow$ Hunter Dominated      Reviewer Dominated $\rightarrow$")
    plt.ylabel("Performance Metrics")
    plt.ylim(0.0, 1.05)  # English comment removed for consistency.
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "Lambda_Sensitivity_F1.png")
    plt.savefig(save_path, dpi=1000)
    print(f"Saved plot to: {save_path}")
    print(f"Stats at Lambda={SELECTED_LAMBDA}: Recall={my_rec:.4f}, Prec={my_prec:.4f}, F1={my_score:.4f}")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    main()