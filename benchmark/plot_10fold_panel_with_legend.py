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
import matplotlib.patches as mpatches
import seaborn as sns

# English comment: see script logic.
# English comment: see script logic.
BASE_DIR = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3"
# English comment: see script logic.
OUTPUT_DIR = "./modality_plots"
# English comment: see script logic.
OUTPUT_FILENAME = "Figure_2_Y_Modality_Importance_Full_Panel.png"

# English comment: see script logic.
MODALITY_NAMES = ['Ligand', 'Interaction', 'ESM2', 'PLIF', 'MaSIF_Global']

# English comment: see script logic.
Y_LIM = (0.18, 0.22)

# English comment: see script logic.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# English comment: see script logic.
def load_fold_data(fold_idx):
    """读取单折数据"""
    pkl_path = os.path.join(BASE_DIR, f"fold_{fold_idx}", "raw_eval_results.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            return data['all_gate_weights'] # (N, 5)
    else:
        print(f"[Warning] Fold {fold_idx} missing!")
        return None

def plot_on_axis(ax, weights, title, palette, is_summary=False):
    """在指定的 Ax 上绘图"""
    df = pd.DataFrame(weights, columns=MODALITY_NAMES)
    df_melt = df.melt(var_name='Modality', value_name='Weight')
    
    # English comment: see script logic.
    sns.boxplot(x='Modality', y='Weight', data=df_melt, ax=ax,
                palette=palette, showfliers=False, width=0.7, linewidth=1.5)
    
    # English comment: see script logic.
    alpha_val = 0.005 if is_summary else 0.05
    sns.stripplot(x='Modality', y='Weight', data=df_melt, ax=ax,
                  color=".25", alpha=alpha_val, size=1.5, jitter=True)
    
    # English comment: see script logic.
    title_color = "#000000" if is_summary else "black"
    title_text = title if is_summary else f"Fold {title}"
    
    ax.set_title(title_text, fontsize=18, fontweight='bold', color=title_color, pad=10)
    ax.set_ylim(Y_LIM)
    ax.set_xlabel('')
    ax.set_ylabel('')  # English comment removed for consistency.
    
    # English comment: see script logic.
    ax.set_xticklabels(MODALITY_NAMES, rotation=15, ha='right', fontsize=13)
    
    # English comment: see script logic.
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

def draw_legend_panel(ax, palette):
    """在第12个格子里画漂亮的图例"""
    ax.axis('off')  # English comment removed for consistency.
    
    # English comment: see script logic.
    legend_handles = []
    for i, name in enumerate(MODALITY_NAMES):
        # English comment: see script logic.
        color = palette[i]
        patch = mpatches.Patch(color=color, label=name)
        legend_handles.append(patch)
    
    # English comment: see script logic.
    ax.legend(handles=legend_handles, 
              title="Modality Legend",
              title_fontsize=22,  # English comment removed for consistency.
              fontsize=18,  # English comment removed for consistency.
              loc='center',
              frameon=True, 
              fancybox=True, 
              edgecolor='black',
              facecolor='#f9f9f9',
              shadow=True,
              borderpad=1.0,  # English comment removed for consistency.
              labelspacing=0.8,  # English comment removed for consistency.
              handlelength=2.0,  # English comment removed for consistency.
              handleheight=1.0  # English comment removed for consistency.
             )

# English comment: see script logic.
def main():
    print("Initializing 3x4 Layout...")
    
    # English comment: see script logic.
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()  # English comment removed for consistency.
    
    # English comment: see script logic.
    # English comment: see script logic.
    palette = sns.color_palette("Set3", n_colors=5)
    
    all_weights_list = []
    
    # English comment: see script logic.
    for i in range(10):
        fold_idx = i + 1
        ax = axes[i]
        
        print(f"Plotting Fold {fold_idx}...")
        weights = load_fold_data(fold_idx)
        
        if weights is not None:
            all_weights_list.append(weights)
            plot_on_axis(ax, weights, str(fold_idx), palette, is_summary=False)
        else:
            ax.text(0.5, 0.5, "Missing Data", ha='center')
            ax.axis('off')
            
    # English comment: see script logic.
    print("Plotting CV Summary...")
    if all_weights_list:
        combined_weights = np.concatenate(all_weights_list, axis=0)
        plot_on_axis(axes[10], combined_weights, "CV Summary (All Folds)", palette, is_summary=True)
    
    # English comment: see script logic.
    print("Drawing Legend...")
    draw_legend_panel(axes[11], palette)
    
    # English comment: see script logic.
    
    # English comment: see script logic.
    # English comment: see script logic.
    for i, ax in enumerate(axes):
        if i % 4 == 0:
            ax.set_ylabel('Attention Weight', fontsize=16, fontweight='bold')
        else:
            ax.set_yticklabels([])  # English comment removed for consistency.
    
    # English comment: see script logic.
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # English comment removed for consistency.
    
    # English comment: see script logic.
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    print(f"\n✅ Full panel saved to: {save_path}")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    main()
