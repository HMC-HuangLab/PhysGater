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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# English comment: see script logic.
# English comment: see script logic.
PKL_FILE = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3/cv_summary/cv_summary_raw_data.pkl"
# English comment: see script logic.
# PKL_FILE = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3/fold_1/raw_eval_results.pkl"

OUTPUT_DIR = "./interpretability"
# =======================================

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
sns.set_theme(style="whitegrid", font_scale=1.2)

def main():
    print(f"Loading {PKL_FILE}...")
    try:
        with open(PKL_FILE, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {PKL_FILE}")
        return

    # English comment: see script logic.
    weights = None
    if isinstance(data, dict):
        if 'all_gate_weights' in data: 
            weights = data['all_gate_weights']
        elif 'gate_weights' in data:   
            weights = data['gate_weights']
        elif 'cv_gate_weights' in data: 
             pass

    # English comment: see script logic.
    if weights is None:
        print("Warning: Could not find weights in pickle. Trying to load Fold 1 specific file...")
        fold1_path = os.path.join(os.path.dirname(os.path.dirname(PKL_FILE)), "fold_1/raw_eval_results.pkl")
        if os.path.exists(fold1_path):
            with open(fold1_path, 'rb') as f:
                data = pickle.load(f)
                weights = data['all_gate_weights']
        else:
            print(f"Error: Could not find {fold1_path}")
            return

    print(f"Weights shape: {weights.shape}") 
    
    # English comment: see script logic.
    modality_names = [
        "Ligand (FP)", 
        "Interaction (Attn)", 
        "Sequence (ESM)", 
        "Fingerprint (PLIF)", 
        "Pocket (MaSIF)"
    ]
    
    # English comment: see script logic.
    df = pd.DataFrame(weights, columns=modality_names)
    
    # English comment: see script logic.
    df_melt = df.melt(var_name="Modality", value_name="Attention Weight")
    
    # English comment: see script logic.
    plt.figure(figsize=(12, 7))
    
    # English comment: see script logic.
    # English comment: see script logic.
    sns.boxplot(
        x="Modality", 
        y="Attention Weight", 
        data=df_melt, 
        hue="Modality",  # English comment removed for consistency.
        legend=False,  # English comment removed for consistency.
        palette="Set2", 
        width=0.6, 
        showfliers=False
    )
    
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    sns.pointplot(
        x="Modality", 
        y="Attention Weight", 
        data=df_melt,
        linestyle='none',  # English comment removed for consistency.
        color="black", 
        markers="d", 
        markersize=8,  # English comment removed for consistency.
        errorbar=None, 
        label='Mean'
    )

    plt.title("Modality Contribution Analysis (Path A)", fontsize=16, fontweight='bold')
    plt.ylabel("Gated Attention Weight (0-1)")
    plt.xlabel("")
    plt.ylim(0, 0.6) 
    
    # English comment: see script logic.
    plt.axhline(0.2, color='red', linestyle='--', alpha=0.5, label="Uniform (0.2)")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Modality_Importance_Boxplot.png")
    plt.savefig(save_path, dpi=1000)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    main()