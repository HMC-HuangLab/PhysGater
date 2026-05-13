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
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import glob
from tqdm import tqdm
import warnings

# English comment: see script logic.
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("请确保环境中已安装 RDKit。")

warnings.filterwarnings('ignore')

# ==========================================
# English comment: see script logic.
# ==========================================
class Config:
    # English comment: see script logic.
    test_csv_path = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/new_dekois_ids.csv"
    output_csv_path = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/physgater_cascade_results.csv"
    
    # English comment: see script logic.
    weights_B = "/mnt/data/fpdetec_V2/PathB_Hunter_Corrected_v3"
    weights_A = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V2"
    
    # English comment: see script logic.
    esm2_root = "/mnt/data/fpdetec_V2/new_esm2_feat"
    masif_root = "/mnt/data/fpdetec_V2/pocket_256_masif_feat"
    plif_root = "/mnt/data/fpdetec_V2/Dataset_PLIF_Flat"
    
    # English comment: see script logic.
    morgan_cache = "/mnt/data/fpdetec_V2/cascade_morgan_cache.pkl"
    plif_cache = "/mnt/data/fpdetec_V2/cascade_plif_cache.pkl"

    # English comment: see script logic.
    batch_size = 64
    threshold_B = 0.5  # English comment removed for consistency.
    threshold_A = 0.5  # English comment removed for consistency.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # English comment: see script logic.
    ligand_dim, esm2_dim = 2048, 2560
    masif_dim, masif_patches, plif_dim = 80, 256, 8
    hidden_A, hidden_B = 256, 512
    projection_dim, num_heads = 128, 4

# ==========================================
# English comment: see script logic.
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.net(x))

# English comment: see script logic.
class PathB_Hunter_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ligand_encoder = nn.Sequential(nn.Linear(cfg.ligand_dim, cfg.hidden_B), nn.BatchNorm1d(cfg.hidden_B), nn.ReLU(), nn.Dropout(0.4), ResidualBlock(cfg.hidden_B, 0.4))
        self.esm_encoder = nn.Sequential(nn.Linear(cfg.esm2_dim, cfg.hidden_B), nn.BatchNorm1d(cfg.hidden_B), nn.ReLU(), nn.Dropout(0.4), ResidualBlock(cfg.hidden_B, 0.4))
        self.classifier = nn.Sequential(nn.Linear(cfg.hidden_B * 2, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, ligand, esm):
        l_emb, p_emb = self.ligand_encoder(ligand), self.esm_encoder(esm)
        return self.classifier(torch.cat([l_emb, p_emb], dim=1))

# English comment: see script logic.
class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))

class MaSIFAttentionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ligand_fc = nn.Sequential(nn.Linear(cfg.ligand_dim, cfg.hidden_A), nn.LayerNorm(cfg.hidden_A), nn.ReLU(), nn.Dropout(0.05), ResidualMLP(cfg.hidden_A))
        self.esm_fc = nn.Sequential(nn.Linear(cfg.esm2_dim, cfg.hidden_A), nn.LayerNorm(cfg.hidden_A), nn.ReLU(), ResidualMLP(cfg.hidden_A))
        self.masif_encoder = nn.Sequential(nn.Linear(cfg.masif_dim, cfg.hidden_A), ResidualMLP(cfg.hidden_A))
        self.plif_fc = nn.Sequential(nn.Linear(cfg.plif_dim, 64), nn.ReLU(), nn.Linear(64, cfg.hidden_A), ResidualMLP(cfg.hidden_A))
        self.gate = nn.Sequential(nn.Linear(cfg.hidden_A * 5, cfg.hidden_A), nn.ReLU(), nn.Linear(cfg.hidden_A, 5))
        self.classifier = nn.Sequential(nn.Linear(cfg.hidden_A * 6, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        # English comment: see script logic.
        self.projector = nn.Linear(cfg.hidden_A * 6, cfg.projection_dim)
        self.ligand_ctx = nn.Linear(cfg.hidden_A, cfg.hidden_A)
        self.protein_ctx = nn.Linear(cfg.hidden_A * 3, cfg.hidden_A)

    def forward(self, ligand, esm, masif, plif):
        l_v, e_v, p_v = self.ligand_fc(ligand), self.esm_fc(esm), self.plif_fc(plif)
        m_all = self.masif_encoder(masif)
        m_g = m_all.mean(dim=1)
        i_v = l_v  # English comment removed for consistency.
        stacked = torch.stack([l_v, i_v, e_v, p_v, m_g], dim=1)
        w = torch.softmax(self.gate(stacked.flatten(1)), dim=-1).unsqueeze(-1)
        f_v = (stacked * w).sum(dim=1)
        return self.classifier(torch.cat([f_v, l_v, i_v, e_v, p_v, m_g], dim=1))

# ==========================================
# English comment: see script logic.
# ==========================================
def get_ci(matrix):
    means = matrix.mean(0)
    se = matrix.std(0, ddof=1) / math.sqrt(matrix.shape[0])
    ci_half = 1.96 * se
    return means, ci_half

class CascadeDataset(Dataset):
    def __init__(self, df, morgan_dict, plif_dict, config, stage='B'):
        self.df = df.reset_index(drop=True)
        self.morgan_dict, self.plif_dict, self.cfg, self.stage = morgan_dict, plif_dict, config, stage
        self.pid_col = 'protien_id' if 'protien_id' in df.columns else 'protein_id'
        
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name, pid = str(row['Name']), str(row[self.pid_col]).strip()
        data = {
            'Name': name, 'Target': str(row.get('Target_Name', 'Unknown')),
            'ligand': torch.tensor(self.morgan_dict.get(name, np.zeros(self.cfg.ligand_dim)), dtype=torch.float32),
            'esm': torch.load(os.path.join(self.cfg.esm2_root, f"{pid}.pt"), map_location='cpu').mean(0) if os.path.exists(os.path.join(self.cfg.esm2_root, f"{pid}.pt")) else torch.zeros(self.cfg.esm2_dim)
        }
        if self.stage == 'A':
            data['masif'] = torch.load(os.path.join(self.cfg.masif_root, f"{pid}.pt"), map_location='cpu') if os.path.exists(os.path.join(self.cfg.masif_root, f"{pid}.pt")) else torch.zeros((self.cfg.masif_patches, self.cfg.masif_dim))
            data['plif'] = torch.tensor(self.plif_dict.get(name, np.zeros(self.cfg.plif_dim)), dtype=torch.float32)
        return data

# ==========================================
# English comment: see script logic.
# ==========================================
if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    # English comment: see script logic.
    df_all = pd.read_csv(Config.test_csv_path)
    with open(Config.morgan_cache, 'rb') as f: morgan_data = pickle.load(f)
    
    # English comment: see script logic.
    print(f"[*] 阶段 1: 启动 PathB 粗筛 (10-Fold Ensemble)...")
    models_B = []
    for i in range(10):
        m = PathB_Hunter_Net(Config).to(Config.device)
        m.load_state_dict(torch.load(os.path.join(Config.weights_B, f"fold_{i+1}", "best_model.pth"), map_location=Config.device))
        m.eval(); models_B.append(m)
    
    loader_B = DataLoader(CascadeDataset(df_all, morgan_data, None, Config, stage='B'), batch_size=Config.batch_size)
    probs_B, names_B, targets_B = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader_B, desc="PathB Inference"):
            l, e = batch['ligand'].to(Config.device), batch['esm'].to(Config.device)
            fold_p = [torch.sigmoid(m(l, e)).cpu().numpy().flatten() for m in models_B]
            probs_B.append(np.array(fold_p))
            names_B.extend(batch['Name']); targets_B.extend(batch['Target'])
    
    matrix_B = np.concatenate(probs_B, axis=1)
    means_B, ci_B = get_ci(matrix_B)
    
    # English comment: see script logic.
    df_res_B = pd.DataFrame({
        'Name': names_B, 'Target_Name': targets_B,
        'pathB_Score': means_B,
        'pathB_Label': (means_B >= Config.threshold_B).astype(int),
        'pathB_CI': [f"[{means_B[i]-ci_B[i]:.4f}, {means_B[i]+ci_half:.4f}]" for i, ci_half in enumerate(ci_B)]
    })
    
    # English comment: see script logic.
    actives_only = df_res_B[df_res_B['pathB_Label'] == 1].copy()
    print(f"[*] 阶段 2: PathB 筛出 {len(actives_only)} 个候选分子，启动 PathA 精筛...")
    
    if len(actives_only) > 0:
        with open(Config.plif_cache, 'rb') as f: plif_data = pickle.load(f)
        # English comment: see script logic.
        actives_with_info = actives_only.merge(df_all[['Name', 'protien_id' if 'protien_id' in df_all.columns else 'protein_id']], on='Name')
        
        models_A = []
        for i in range(10):
            m = MaSIFAttentionNet(Config).to(Config.device)
            # English comment: see script logic.
            m.load_state_dict(torch.load(os.path.join(Config.weights_A, f"fold_{i+1}", "best_model.pth"), map_location=Config.device), strict=False)
            m.eval(); models_A.append(m)
            
        loader_A = DataLoader(CascadeDataset(actives_with_info, morgan_data, plif_data, Config, stage='A'), batch_size=Config.batch_size)
        probs_A, names_A = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader_A, desc="PathA Inference"):
                l, e, m, pl = batch['ligand'].to(Config.device), batch['esm'].to(Config.device), batch['masif'].to(Config.device), batch['plif'].to(Config.device)
                fold_p = [torch.sigmoid(model(l, e, m, pl)).cpu().numpy().flatten() for model in models_A]
                probs_A.append(np.array(fold_p))
                names_A.extend(batch['Name'])
        
        matrix_A = np.concatenate(probs_A, axis=1)
        means_A, ci_A = get_ci(matrix_A)
        
        df_res_A = pd.DataFrame({
            'Name': names_A,
            'pathA_Label': (means_A >= Config.threshold_A).astype(int),
            'pathA_CI': [f"[{means_A[i]-ci_half:.4f}, {means_A[i]+ci_half:.4f}]" for i, ci_half in enumerate(ci_A)]
        })
        
        # English comment: see script logic.
        final_df = df_res_B.merge(df_res_A, on='Name', how='left')
    else:
        final_df = df_res_B
        final_df['pathA_Label'] = 0
        final_df['pathA_CI'] = "N/A"

    final_df.to_csv(Config.output_csv_path, index=False)
    print(f"[Done] 级联筛选完成！最终结果保存在: {Config.output_csv_path}")