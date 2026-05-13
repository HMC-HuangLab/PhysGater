# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


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
import seaborn as sns
import pandas as pd
import numpy as np

# English comment: see script logic.
# English comment: see script logic.
N_MOLECULES = 61875

data = {
    "Method": ["Vina-GPU", "PhysGater (Path B)", "PhysGater (Fusion)"],
    
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    "Total_Time_Min": [515.6, 1.2, 15.5], 
    
    # English comment: see script logic.
    "EF1%": [4.5, 22.1, 23.8],
    
    # English comment: see script logic.
    "Hardware": ["4060Ti", "3070", "4090D"]
}

# =============================================================

def main():
    df = pd.read_csv(io.StringIO(csv_data)) if 'csv_data' in locals() else pd.DataFrame(data)
    
    # English comment: see script logic.
    df["Throughput"] = N_MOLECULES / (df["Total_Time_Min"] * 60)
    
    # English comment: see script logic.
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # English comment: see script logic.
    
    # English comment: see script logic.
    color = 'tab:red'
    ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Early Enrichment (EF1%)', color=color, fontsize=14, fontweight='bold')
    
    # English comment: see script logic.
    bars = sns.barplot(x='Method', y='EF1%', data=df, ax=ax1, 
                       palette=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.6, edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 30)  # English comment removed for consistency.
    
    # English comment: see script logic.
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total Screening Time (Minutes) [Log Scale]', color=color, fontsize=14, fontweight='bold')
    
    # English comment: see script logic.
    sns.lineplot(x='Method', y='Total_Time_Min', data=df, ax=ax2, 
                 marker='o', markersize=12, color='blue', linewidth=2, sort=False)
    
    # English comment: see script logic.
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # English comment: see script logic.
    
    # English comment: see script logic.
    for i, p in enumerate(bars.patches):
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., height + 0.5, 
                 f'{height:.1f}', ha="center", color='red', fontweight='bold')
        
    # English comment: see script logic.
    for i, time in enumerate(df['Total_Time_Min']):
        # English comment: see script logic.
        throughput = df["Throughput"][i]
        label = f"{time:.1f} min\n({throughput:.0f} mol/s)"
        ax2.text(i, time * 1.3, label, ha="center", color='blue', fontsize=10, fontweight='bold')

    plt.title(f"Efficiency vs. Accuracy Analysis on MAPK1 (N={N_MOLECULES})", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("Efficiency_Analysis_MAPK1.png", dpi=1000)
    print("Saved: Efficiency_Analysis_MAPK1.png")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    main()