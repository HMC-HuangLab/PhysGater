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
import re

def process_labels(input_file, output_file):
    print(f"正在读取文件: {input_file} ...")
    # English comment: see script logic.
    df = pd.read_csv(input_file)
    
    # English comment: see script logic.
    # English comment: see script logic.
    def parse_name_info(name):
        # English comment: see script logic.
        # English comment: see script logic.
        match = re.search(r'(.+)_(active|decoy)_', name, re.IGNORECASE)
        if match:
            target = match.group(1)
            bg_type = match.group(2).lower()
            # Ground Truth: 1 for active, 0 for decoy
            gt = 1 if bg_type == 'active' else 0
            return target, gt
        else:
            return "Unknown", -1

    print("正在解析分子名称与真实活性标签...")
    # English comment: see script logic.
    parsed_data = df['Name'].apply(parse_name_info)
    df['Target_Name'] = [x[0] for x in parsed_data]
    df['Ground_Truth'] = [x[1] for x in parsed_data]

    # English comment: see script logic.
    if 'Unknown' in df['Target_Name'].values:
        print(f"警告：有 {len(df[df['Target_Name']=='Unknown'])} 行数据无法解析Name格式，已剔除。")
        df = df[df['Target_Name'] != 'Unknown']

    # English comment: see script logic.
    print("正在计算每个靶点的 Top 10% 打分阈值...")
    
    # English comment: see script logic.
    # English comment: see script logic.
    threshold_map = df.groupby('Target_Name')['raw_vina_score'].quantile(0.1).to_dict()
    
    # English comment: see script logic.
    df['Threshold_Top10'] = df['Target_Name'].map(threshold_map)

    # English comment: see script logic.
    print("正在生成四分类标签 (TP, FP, TN, FN)...")

    # English comment: see script logic.
    # English comment: see script logic.
    predicted_positive = df['raw_vina_score'] <= df['Threshold_Top10']
    is_active = df['Ground_Truth'] == 1
    
    conditions = [
        (is_active & predicted_positive),  # English comment removed for consistency.
        (is_active & ~predicted_positive),  # English comment removed for consistency.
        (~is_active & predicted_positive),  # English comment removed for consistency.
        (~is_active & ~predicted_positive)  # English comment removed for consistency.
    ]
    
    choices = ['TP', 'FN', 'FP', 'TN']
    # English comment: see script logic.
    choices_idx = [0, 1, 2, 3] 

    df['Label_4Class'] = np.select(conditions, choices, default='Error')
    df['Label_Idx'] = np.select(conditions, choices_idx, default=-1)

    # English comment: see script logic.
    print("-" * 30)
    print("标签重构统计结果：")
    print(df['Label_4Class'].value_counts())
    print("-" * 30)
    
    # English comment: see script logic.
    # print(df.groupby(['Target_Name', 'Label_4Class']).size().head(20))

    print(f"正在保存结果到: {output_file}")
    # English comment: see script logic.
    output_cols = ['Name', 'smiles', 'Target_Name', 'raw_vina_score', 'Threshold_Top10', 'Ground_Truth', 'Label_4Class', 'Label_Idx', 'protein_id']
    # English comment: see script logic.
    
    df.to_csv(output_file, index=False, columns=[c for c in output_cols if c in df.columns])
    print("处理完成！")

# English comment: see script logic.
# English comment: see script logic.
if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    input_csv = "/mnt/data/fpdetec_V2/final_with_thresholds/new_combined_dataset.csv"  # English comment removed for consistency.
    output_csv = "/mnt/data/fpdetec_V2/final_with_thresholds/new_train_data_relabelled.csv"  # English comment removed for consistency.
    
    # English comment: see script logic.
    process_labels(input_csv, output_csv)
    # English comment: see script logic.