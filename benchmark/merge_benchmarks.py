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
import os
import glob

# ==========================================
# English comment: see script logic.
# ==========================================
INPUT_DIR = "/mnt/data/fpdetec_V2/bench_results/"
OUTPUT_DIR = "/mnt/data/fpdetec_V2/bench_results/merged_benchmarks/"
DATASETS = ["DUD-E", "LIT-PCBA", "DEKOIS2"]

def generate_lit_pcba_key(name):
    """专门针对 LIT-PCBA 的逻辑：提取首部和尾部"""
    if pd.isna(name):
        return None
    parts = str(name).strip().split('_')
    
    # English comment: see script logic.
    # English comment: see script logic.
    # PhysGater: parts[0]=Target, parts[-2]=active/decoy, parts[-1]=num
    # English comment: see script logic.
    if len(parts) >= 3:
        target = parts[0].lower()
        m_type = parts[-2].lower()
        num = parts[-1]
        return f"{target}_{m_type}_{num}"
    return str(name).lower()  # English comment removed for consistency.

def clean_data_robust(df, ds_name):
    """针对不同数据集采用不同的清洗策略"""
    df = df.dropna(subset=['Name']).copy()
    df['Name'] = df['Name'].astype(str).str.strip()
    
    if ds_name == "LIT-PCBA":
        # English comment: see script logic.
        df['match_key'] = df['Name'].apply(generate_lit_pcba_key)
    else:
        # English comment: see script logic.
        df['match_key'] = df['Name'].str.lower()
    
    return df

def merge_by_physgater_v3():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for ds in DATASETS:
        print(f"\n" + "="*40)
        print(f"[*] 正在处理数据集: {ds}")
        
        # English comment: see script logic.
        phys_pattern = os.path.join(INPUT_DIR, f"*Physgater*{ds}*.csv")
        phys_files = glob.glob(phys_pattern)
        if not phys_files: continue
            
        base_df = pd.read_csv(phys_files[0])
        base_df = clean_data_robust(base_df, ds)
        initial_count = len(base_df)
        
        # --- 2. Vina-GPU ---
        if 'raw_vina_score' in base_df.columns:
            base_df['Vina-GPU_score'] = base_df['raw_vina_score']

        # English comment: see script logic.
        other_models = [
            ("PLANET", "PLANET_score", "PLANET_score"),
            ("Karmadock", "karma_score", "Karmadock_score"),
            ("RF-Score-VS", "rf_score", "RF-Score-VS_score")
        ]

        for model_key, old_col, new_col in other_models:
            pattern = os.path.join(INPUT_DIR, f"*{model_key}*{ds}*.csv")
            files = glob.glob(pattern)
            
            if files:
                other_df = pd.read_csv(files[0])
                if old_col in other_df.columns:
                    other_df = clean_data_robust(other_df, ds)
                    
                    # English comment: see script logic.
                    temp_df = other_df[['match_key', old_col]].copy()
                    temp_df = temp_df.rename(columns={old_col: new_col})
                    # English comment: see script logic.
                    temp_df = temp_df.drop_duplicates(subset=['match_key'])
                    
                    # English comment: see script logic.
                    base_df = pd.merge(base_df, temp_df, on='match_key', how='left')
                    
                    # English comment: see script logic.
                    matched = base_df[new_col].notna().sum()
                    print(f"   --> {model_key}: 匹配成功 {matched}/{initial_count} ({matched/initial_count:.1%})")

        # English comment: see script logic.
        if 'match_key' in base_df.columns:
            base_df = base_df.drop(columns=['match_key'])
        
        output_name = os.path.join(OUTPUT_DIR, f"{ds}_unified_comparison.csv")
        base_df.to_csv(output_name, index=False)
        print(f"[OK] {ds} 处理完成。")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    merge_by_physgater_v3()