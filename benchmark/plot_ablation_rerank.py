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
from sklearn.metrics import roc_auc_score, precision_score

# English comment: see script logic.
class Config:
    DATASETS = {
        "DUD-E": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/DUD-E_unified_comparison.csv",
        "DEKOIS": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/DEKOIS2_unified_comparison.csv",
        "LIT-PCBA": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/LIT-PCBA_unified_comparison.csv"
    }
    
    # English comment: see script logic.
    VARIANTS = {
        "Path B (Hunter)":   {"col": "Score_Hunter_B",   "color": "#ff7f0e"},
        "Path A (Reviewer)": {"col": "Score_Reviewer_A", "color": "#1f77b4"},
        "PhysGater (Fusion)":{"col": "PhysGater_Score",  "color": "#d62728"},
    }
    
    OUTPUT_DIR = "./ablation_consistent"
    DPI = 1000

# English comment: see script logic.
    FONT_SIZE_TITLE = 20  # English comment removed for consistency.
    FONT_SIZE_AXIS_TITLE = 18  # English comment removed for consistency.
    FONT_SIZE_TICK = 15  # English comment removed for consistency.
    FONT_SIZE_LEGEND = 16  # English comment removed for consistency.
    FONT_SIZE_BAR_VAL = 12  # English comment removed for consistency.

if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)

# English comment: see script logic.
sns.set_theme(style="whitegrid", rc={
    "axes.labelsize": Config.FONT_SIZE_AXIS_TITLE,
    "xtick.labelsize": Config.FONT_SIZE_TICK,
    "ytick.labelsize": Config.FONT_SIZE_TICK,
    "legend.fontsize": Config.FONT_SIZE_LEGEND,
    "font.family": "sans-serif"  # English comment removed for consistency.
})

def calc_metrics_at_top_k(y_true, y_scores, percent=0.01):
    n_total = len(y_true)
    n_actives = sum(y_true)
    n_negatives = n_total - n_actives
    n_top = max(1, int(n_total * percent))
    
    idx = np.argsort(y_scores)[::-1][:n_top]
    tp = sum(y_true[idx])
    fp = n_top - tp
    
    ef = (tp / n_top) / (n_actives / n_total) if n_actives > 0 else 0
    precision = tp / n_top
    fpr = fp / n_negatives if n_negatives > 0 else 0
    
    return ef, precision, fpr

def main():
    summary = []
    
    print("Calculating metrics...")
    for name, path in Config.DATASETS.items():
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        y_true = df['Ground_Truth'].values
        
        for v_name, cfg in Config.VARIANTS.items():
            if cfg['col'] not in df.columns: continue
            
            scores = df[cfg['col']].values
            scores = np.nan_to_num(scores, nan=-999)
            
            # English comment: see script logic.
            auc_val = roc_auc_score(y_true, scores)
            ef1, prec1, fpr1 = calc_metrics_at_top_k(y_true, scores, 0.01)
            
            # English comment: see script logic.
            summary.append({
                "Dataset": name,
                "Variant": v_name,
                "AUC": auc_val * 100,           # 0.75 -> 75.0
                "EF1%": ef1,  # English comment removed for consistency.
                                                # English comment: see script logic.
                "Precision@1%": prec1 * 100,    # 0.4 -> 40.0
                "FPR@1%": fpr1 * 100            # 0.005 -> 0.5
            })

    df_res = pd.DataFrame(summary)
    
    # English comment: see script logic.
    metrics_config = [
        {"col": "AUC", "title": "Global Ranking (AUC)", "ylabel": "ROC-AUC (%)", "fmt": "%.1f", "lower": False},
        # English comment: see script logic.
        # English comment: see script logic.
        # English comment: see script logic.
        # English comment: see script logic.
        # English comment: see script logic.
        {"col": "EF1%", "title": "Early Enrichment (EF1%)", "ylabel": "EF Score", "fmt": "%.1f", "lower": False},
        
        {"col": "Precision@1%", "title": "Hit Rate (Precision)", "ylabel": "Precision (%)", "fmt": "%.1f", "lower": False},
        {"col": "FPR@1%", "title": "False Positive Rate", "ylabel": "FPR (%)", "fmt": "%.2f", "lower": True}
    ]
    
    print("Generating Plots...")
    for m in metrics_config:
        plt.figure(figsize=(10, 7))  # English comment removed for consistency.
        
        # English comment: see script logic.
        ax = sns.barplot(
            data=df_res, 
            x="Dataset", 
            y=m['col'], 
            hue="Variant",
            palette={k:v['color'] for k,v in Config.VARIANTS.items()},
            edgecolor="black", 
            linewidth=1.2
        )
        
        # English comment: see script logic.
        # suffix = " (↓)" if m['lower'] else " (↑)"
        plt.title(f"{m['title']}", fontsize=Config.FONT_SIZE_TITLE, fontweight='bold', pad=20)
        plt.ylabel(m['ylabel'], fontsize=Config.FONT_SIZE_AXIS_TITLE, fontweight='bold', labelpad=10)
        plt.xlabel("Dataset", fontsize=Config.FONT_SIZE_AXIS_TITLE, fontweight='bold', labelpad=10)
        
        # English comment: see script logic.
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.15)
        
        # English comment: see script logic.
        for container in ax.containers:
            ax.bar_label(container, fmt=m['fmt'], padding=3, fontsize=Config.FONT_SIZE_BAR_VAL)

        # English comment: see script logic.
        # English comment: see script logic.
        # English comment: see script logic.
        # English comment: see script logic.
        plt.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.18), 
            ncol=3, 
            frameon=False,
            fontsize=Config.FONT_SIZE_LEGEND,
            title=None  # English comment removed for consistency.
        )
        
        plt.tight_layout()
        
        # English comment: see script logic.
        safe_name = m['col'].replace("%", "").replace("@", "_")
        save_path = os.path.join(Config.OUTPUT_DIR, f"Ablation_Final_{safe_name}.png")
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')  # English comment removed for consistency.
        print(f"  -> Saved: {save_path}")

    print("\n✅ All final plots generated successfully!")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    main()
