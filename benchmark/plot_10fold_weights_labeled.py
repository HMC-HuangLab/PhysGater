# -*- coding: utf-8 -*-
import os


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
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# English comment: see script logic.
# English comment: see script logic.
BASE_DIR = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3"
OUTPUT_DIR = "./modality_plots"  # English comment removed for consistency.

# English comment: see script logic.
# English comment: see script logic.
MODALITY_NAMES = ['Ligand', 'Interaction', 'ESM2', 'PLIF', 'MaSIF_Global']

# English comment: see script logic.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def plot_single_fold(fold_idx):
    fold_dir = os.path.join(BASE_DIR, f"fold_{fold_idx}")
    pkl_path = os.path.join(fold_dir, "raw_eval_results.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"[Warning] Fold {fold_idx} data not found: {pkl_path}")
        return

    print(f"Processing Fold {fold_idx}...")
    
    # English comment: see script logic.
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    weights = data['all_gate_weights'] # Shape: (N, 5)
    
    # English comment: see script logic.
    df_weights = pd.DataFrame(weights, columns=MODALITY_NAMES)
    df_melt = df_weights.melt(var_name='Modality', value_name='Weight')
    
    # English comment: see script logic.
    plt.figure(figsize=(8, 6))  # English comment removed for consistency.
    
    # English comment: see script logic.
    sns.boxplot(x='Modality', y='Weight', data=df_melt, 
                palette="Set3", showfliers=False, width=0.6, linewidth=1.5)
    
    # English comment: see script logic.
    sns.stripplot(x='Modality', y='Weight', data=df_melt, 
                  color=".25", alpha=0.05, size=2, jitter=True)
    
    # English comment: see script logic.
    plt.title(f"Modality Importance (Fold {fold_idx})", fontsize=16, fontweight='bold', pad=12)
    plt.ylabel('Attention Weight', fontsize=14)
    plt.xlabel('')  # English comment removed for consistency.
    plt.ylim(0.18, 0.22)  # English comment removed for consistency.
    
    # English comment: see script logic.
    plt.tight_layout()
    
    # English comment: see script logic.
    save_path = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_modality_importance_labeled.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    for i in range(1, 11):
        plot_single_fold(i)
    
    print(f"\n✅ All 10 plots saved to: {OUTPUT_DIR}")
    print("Next Step: Use the 'merge_plots.py' script to combine them into one figure.")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    main()