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
from sklearn.metrics import roc_auc_score

# English comment: see script logic.
class Config:
    # English comment: see script logic.
    DATASETS = {
        "DUD-E": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/DUD-E_unified_comparison.csv",
        "DEKOIS": "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/DEKOIS2_unified_comparison.csv"
    }
    
    OUTPUT_DIR = "./family_analysis"
    
    # English comment: see script logic.
    MODELS = {
        "PhysGater (Ours)": {"col": "PhysGater_Score", "sign": 1, "color": "#d62728", "style": "-"},
        "PLANET":           {"col": "PLANET_score",    "sign": 1, "color": "#ff7f0e", "style": "--"},
        "RF-Score-VS":      {"col": "RF-Score-VS_score",        "sign": 1, "color": "#2ca02c", "style": "--"},
        "Vina-GPU":         {"col": "Vina-GPU_score",      "sign": -1, "color": "#1f77b4", "style": ":"},
        "KarmaDock":        {"col": "Karmadock_score",     "sign": 1, "color": "#9467bd", "style": ":"}
    }
    
    # English comment: see script logic.
    TARGET_COL_CANDIDATES = ['Target', 'System', 'target', 'protein', 'prot_id', 'Target_Name']

    DPI=1000

# English comment: see script logic.
# English comment: see script logic.
FAMILY_MAPPING = {
    # English comment: see script logic.
    'cdk2': 'Kinase', 'src': 'Kinase', 'egfr': 'Kinase', 'vegfr2': 'Kinase', 'vegfr1': 'Kinase',
    'p38': 'Kinase', 'tk': 'Kinase', 'pdgfrb': 'Kinase', 'fgfr1': 'Kinase', 'braf': 'Kinase',
    'itk': 'Kinase', 'rock-1': 'Kinase', 'mk2': 'Kinase', 'akt1': 'Kinase', 'pim': 'Kinase',
    'pim-1kinase': 'Kinase', 'pim2': 'Kinase', 'prkcq': 'Kinase', 'jnk2': 'Kinase', 'jnk3': 'Kinase',
    'aurka': 'Kinase', 'gsk3b': 'Kinase', 'erbb2': 'Kinase', 'tie2': 'Kinase', 'lck': 'Kinase',
    'jak3': 'Kinase', 'ephrinb4': 'Kinase', 'igf-1r': 'Kinase', 'pi3kg': 'Kinase',
    
    # English comment: see script logic.
    'adrb2': 'GPCR', 'aa2ar': 'GPCR', 'a2a': 'GPCR', 'cxcr4': 'GPCR',
    
    # English comment: see script logic.
    'ar': 'Nuclear Receptor', 'gr': 'Nuclear Receptor', 'pr': 'Nuclear Receptor', 
    'er-agonist': 'Nuclear Receptor', 'er-antagonist': 'Nuclear Receptor', 'er-beta': 'Nuclear Receptor',
    'ppar': 'Nuclear Receptor', 'pparg': 'Nuclear Receptor', 'ppar-a': 'Nuclear Receptor',
    'rxr': 'Nuclear Receptor', 'rxr-a': 'Nuclear Receptor', 'mr': 'Nuclear Receptor',
    
    # English comment: see script logic.
    'fxa': 'Protease', 'trypsin': 'Protease', 'thrombin': 'Protease', 'hivpr': 'Protease', 'hiv-1pr': 'Protease',
    'ace': 'Protease', 'ace2': 'Protease', 'tpa': 'Protease', 'upa': 'Protease', 
    'ctsk': 'Protease', 'mmp2': 'Protease', 'adam17': 'Protease', 'dpp4': 'Protease',
    
    # --- Others (Enzymes, etc.) ---
    'pnp': 'Enzyme/Other', 'cox1': 'Enzyme/Other', 'cox2': 'Enzyme/Other', 
    'comt': 'Enzyme/Other', 'na': 'Enzyme/Other', 'ache': 'Enzyme/Other', 
    'hmga': 'Enzyme/Other', 'pde5': 'Enzyme/Other', 'pde4b': 'Enzyme/Other',
    'parp': 'Enzyme/Other', 'parp-1': 'Enzyme/Other', 'hsp90': 'Enzyme/Other', 
    'dhfr': 'Enzyme/Other', 'sahh': 'Enzyme/Other', 'ampc': 'Enzyme/Other', 
    'inha': 'Enzyme/Other', 'gart': 'Enzyme/Other', 'hivrt': 'Enzyme/Other', 
    'alr2': 'Enzyme/Other', 'gpb': 'Enzyme/Other', 'ada': 'Enzyme/Other',
    'bcl2': 'Enzyme/Other', 'fkbp1a': 'Enzyme/Other', '11hsd': 'Enzyme/Other',
    'gba': 'Enzyme/Other', 'cyp2a6': 'Enzyme/Other', 'mdm2': 'Enzyme/Other', 
    'hdac2': 'Enzyme/Other', 'hdac8': 'Enzyme/Other', 'ts': 'Enzyme/Other', 
    '17b-hsd1': 'Enzyme/Other', 'kif11': 'Enzyme/Other'
}

if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)

# English comment: see script logic.
sns.set_theme(style="whitegrid", rc={
    "axes.labelsize": 20,  # English comment removed for consistency.
    "xtick.labelsize": 16,  # English comment removed for consistency.
    "ytick.labelsize": 16,  # English comment removed for consistency.
    "legend.fontsize": 16,  # English comment removed for consistency.
    "font.family": "sans-serif"
})

def clean_target_name(raw_name):
    s = str(raw_name).strip().lower()
    if s.endswith("_a"): s = s[:-2]
    if '-' in s:
        parts = s.split('-')
        if len(parts[0]) == 4 and parts[0][0].isdigit(): s = "-".join(parts[1:])
    return s

def get_family(raw_name):
    clean_name = clean_target_name(raw_name)
    if clean_name in FAMILY_MAPPING: return FAMILY_MAPPING[clean_name]
    for key, fam in FAMILY_MAPPING.items():
        if key == clean_name: return fam
    
    if 'cox' in clean_name: return 'Enzyme/Other'
    if 'pde' in clean_name: return 'Enzyme/Other'
    if 'hdac' in clean_name: return 'Enzyme/Other'
    if 'jak' in clean_name: return 'Kinase'
    if 'casp' in clean_name: return 'Protease'
    if 'cyp' in clean_name: return 'Enzyme/Other'
    
    return "Other"

def calc_metrics_at_top_k(y_true, y_scores, percent=0.01):
    n_total = len(y_true); n_actives = sum(y_true)
    n_negatives = n_total - n_actives
    if n_actives == 0 or n_negatives == 0: return 0.0, 0.0
    
    n_top = max(1, int(n_total * percent))
    sorted_idx = np.argsort(y_scores)[::-1][:n_top]
    
    tp = sum(y_true[sorted_idx])
    fp = n_top - tp
    
    ef = (tp / n_top) / (n_actives / n_total)
    fpr = fp / n_negatives
    
    return ef, fpr

def draw_family_barplot(df, metric, title, filename, ylabel, fmt='%.2f'):
    # English comment: see script logic.
    plt.figure(figsize=(11, 8.5))
    
    hue_order = [m for m in Config.MODELS.keys() if m in df['Model'].unique()]
    palette = {k: v['color'] for k,v in Config.MODELS.items()}
    
    ax = sns.barplot(
        data=df, 
        x="Family", 
        y=metric, 
        hue="Model", 
        hue_order=hue_order, 
        palette=palette,
        edgecolor="black", 
        linewidth=1.2, 
        errorbar=None
    )
    
    # English comment: see script logic.
    plt.title(title, fontsize=22, fontweight='bold', pad=20)
    plt.xlabel("Target Family", fontsize=20, fontweight='bold', labelpad=12)
    plt.ylabel(ylabel, fontsize=20, fontweight='bold', labelpad=12)
    
    # English comment: see script logic.
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] * 1.15)
    
    # English comment: see script logic.
    for container in ax.containers:
        # English comment: see script logic.
        # English comment: see script logic.
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=12, rotation=0)

    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.18), 
        ncol=3, 
        frameon=False,
        fontsize=16,
        title=None,
        columnspacing=1.5
    )
    
    plt.tight_layout()
    
    save_path = os.path.join(Config.OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def main():
    all_results = []
    print("Starting Multi-Model Family Analysis...")
    
    for ds_name, csv_path in Config.DATASETS.items():
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        
        target_col = None
        for col in Config.TARGET_COL_CANDIDATES:
            if col in df.columns:
                target_col = col; break
        if not target_col: continue
            
        df['Family'] = df[target_col].apply(get_family)
        
        for fam in df['Family'].unique():
            if fam == "Other": continue
            
            sub_df = df[df['Family'] == fam]
            y_true = sub_df['Ground_Truth'].values
            if len(y_true) < 10 or sum(y_true) == 0: continue
            
            for model_name, cfg in Config.MODELS.items():
                if cfg['col'] not in df.columns: continue
                
                scores = sub_df[cfg['col']].values * cfg['sign']
                scores = np.nan_to_num(scores, nan=-999)
                
                auc_val = roc_auc_score(y_true, scores)
                ef1, fpr1 = calc_metrics_at_top_k(y_true, scores, 0.01)
                ef5, fpr5 = calc_metrics_at_top_k(y_true, scores, 0.05)
                
                all_results.append({
                    "Dataset": ds_name,
                    "Family": fam,
                    "Model": model_name,
                    # English comment: see script logic.
                    "AUC": auc_val * 100,
                    "EF1%": ef1,  # English comment removed for consistency.
                    "EF5%": ef5,
                    "FPR@Top1%": fpr1 * 100,
                    "FPR@Top5%": fpr5 * 100
                })

    if not all_results: return
    res_df = pd.DataFrame(all_results)
    
    print("Generating Plots...")
    
    # English comment: see script logic.
    draw_family_barplot(res_df, "FPR@Top1%", "Noise Level (FPR @ Top 1%) ", "Family_FPR1.png", "FPR (%)", fmt='%.2f')
    draw_family_barplot(res_df, "FPR@Top5%", "Noise Level (FPR @ Top 5%) ", "Family_FPR5.png", "FPR (%)", fmt='%.2f')
    
    # English comment: see script logic.
    draw_family_barplot(res_df, "EF1%", "Early Enrichment (EF1%) ", "Family_EF1.png", "EF Score", fmt='%.1f')
    
    # English comment: see script logic.
    draw_family_barplot(res_df, "AUC", "Global Ranking (ROC-AUC)", "Family_AUC.png", "ROC-AUC (%)", fmt='%.1f')

    print(f"\n✅ All family analysis plots saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    main()