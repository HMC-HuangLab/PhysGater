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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# English comment: see script logic.
class Config:
    DATASETS = {
        "DUD-E": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/DUD-E_unified_comparison.csv",
        "DEKOIS": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/DEKOIS2_unified_comparison.csv",
        "LIT-PCBA": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/LIT-PCBA_unified_comparison.csv"
    }
    
    
    # English comment: see script logic.
    VARIANTS = {
        "Path A (Reviewer)": {"col": "Score_Reviewer_A", "color": "#1f77b4", "style": "--"},
        "Path B (Hunter)":   {"col": "Score_Hunter_B",   "color": "#ff7f0e", "style": "--"},
        "PhysGater (Fusion)":{"col": "PhysGater_Score",  "color": "#d62728", "style": "-"},
    }
    
    OUTPUT_DIR = "/mnt/data/fpdetec_V2/bench_results/ablation"

if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)
sns.set_theme(style="whitegrid", font_scale=1.2)

def calc_ef(y_true, y_scores, percent=0.01):
    n_actives = sum(y_true)
    n_top = int(len(y_true) * percent)
    if n_actives == 0 or n_top == 0: return 0.0
    sorted_idx = np.argsort(y_scores)[::-1][:n_top]
    actives_found = sum(y_true[sorted_idx])
    return (actives_found / n_top) / (n_actives / len(y_true))

def main():
    summary_list = []

    for ds_name, csv_path in Config.DATASETS.items():
        print(f"Processing {ds_name}...")
        if not os.path.exists(csv_path): continue
        
        df = pd.read_csv(csv_path)
        y_true = df['Ground_Truth'].values
        
        # English comment: see script logic.
        plt.figure(figsize=(8, 6))
        
        for v_name, cfg in Config.VARIANTS.items():
            if cfg['col'] not in df.columns: continue
            
            scores = df[cfg['col']].values
            # English comment: see script logic.
            scores = np.nan_to_num(scores, nan=0)
            
            # English comment: see script logic.
            auc_val = roc_auc_score(y_true, scores)
            ef1 = calc_ef(y_true, scores, 0.01)
            
            summary_list.append({
                "Dataset": ds_name,
                "Variant": v_name,
                "AUC": auc_val,
                "EF1%": ef1
            })
            
            # English comment: see script logic.
            fpr, tpr, _ = roc_curve(y_true, scores)
            plt.plot(fpr, tpr, label=f"{v_name} (AUC={auc_val:.3f})", 
                     color=cfg['color'], linestyle=cfg['style'], linewidth=2)
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{ds_name}: Ablation Study ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f"Ablation_ROC_{ds_name}.png"), dpi=300)
        plt.close()

    # English comment: see script logic.
    df_sum = pd.DataFrame(summary_list)
    
    # AUC Bar
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sum, x="Dataset", y="AUC", hue="Variant", 
                palette={k:v['color'] for k,v in Config.VARIANTS.items()})
    plt.title("Ablation: Impact of Fusion on AUC")
    plt.ylim(0.5, 1.0)  # English comment removed for consistency.
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "Ablation_Bar_AUC.png"), dpi=300)
    
    # EF Bar
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sum, x="Dataset", y="EF1%", hue="Variant",
                palette={k:v['color'] for k,v in Config.VARIANTS.items()})
    plt.title("Ablation: Impact of Fusion on Early Enrichment (EF1%)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "Ablation_Bar_EF1.png"), dpi=300)
    
    print("Ablation plots generated.")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    main()
