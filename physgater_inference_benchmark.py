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
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ==========================================
# English comment: see script logic.
# ==========================================
class Config:
    # English comment: see script logic.
    BENCHMARK_CSV = "/mnt/data/multitaskfp/LTI-PBCA/merged_results/final_lti_pbca_cleaned.csv"  # English comment removed for consistency.
    
    # English comment: see script logic.
    BENCHMARK_MORGAN_CACHE = "/mnt/data/multitaskfp/LTI-PBCA/merged_results/LTI_PBCA_morgan_2048_cache.pkl"
    BENCHMARK_PLIF_CACHE = "/mnt/data/multitaskfp/LTI-PBCA/merged_results/LTI-PBCA_plif_cache.pkl"
    
    BENCHMARK_DATA_ROOT = "/mnt/data/multitaskfp/LTI-PBCA/merged_results/LTI-PBCA_PLIF_Flat/"

    # English comment: see script logic.
    ESM2_ROOT = "/mnt/data/multitaskfp/LTI-PBCA/esm2_feat/esm2_256_feat/"    
    MASIF_ROOT = "/mnt/data/multitaskfp/LTI-PBCA/masif_feat_pt/256dim_pt" 
    
    # English comment: see script logic.
    PATH_A_ROOT = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3"
    PATH_B_ROOT = "/mnt/data/fpdetec_V2/PathB_Hunter_Corrected_v3"
    
    # English comment: see script logic.
    THRESH_B_HUNTER = 0.37  # English comment removed for consistency.
    THRESH_A_REVIEWER = 0.40  # English comment removed for consistency.

    # English comment: see script logic.
    LAMBDA_A = 0.14
    LAMBDA_B = 0.86
    FINAL_THRESHOLD = 0.4600
    
    # English comment: see script logic.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # English comment: see script logic.
    OUTPUT_FILE = "physgater_lti-pcba_results.csv"

# ==========================================
# English comment: see script logic.
# ==========================================
class ConfigA:
    ligand_dim = 2048; esm2_dim = 2560; masif_dim = 80; masif_patches = 256
    plif_dim = 8; hidden_dim = 256; projection_dim = 128; num_heads = 4; bit_dropout_prob = 0.0

class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(query_dim, embed_dim); self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim); self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, query, key_value):
        bsz, seq_len, _ = key_value.size()
        q = self.q_proj(query).view(bsz, self.num_heads, 1, self.head_dim)
        k = self.k_proj(key_value).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(attn, dim=-1)
        context = torch.matmul(weights, v).squeeze(2).transpose(1, 2).reshape(bsz, self.embed_dim)
        return self.out_proj(context)

class MasifPatchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, patches, groups=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(ResidualMLP(hidden_dim), ResidualMLP(hidden_dim))
        self.global_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU())
    def forward(self, masif):
        x = self.blocks(self.input_proj(masif))
        return x, self.global_fusion(torch.cat([x.mean(dim=1), x.amax(dim=1), x.mean(dim=1)], dim=-1))

class GatedFusion(nn.Module):
    def __init__(self, feature_dim, num_modalities):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(feature_dim * num_modalities, feature_dim), nn.ReLU(), nn.Linear(feature_dim, num_modalities))
    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        weights = torch.softmax(self.gate(stacked.flatten(start_dim=1)), dim=-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=1)

class MaSIFAttentionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ligand_fc = nn.Sequential(nn.Linear(cfg.ligand_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.bit_dropout_prob), ResidualMLP(cfg.hidden_dim), ResidualMLP(cfg.hidden_dim))
        self.masif_encoder = MasifPatchEncoder(cfg.masif_dim, cfg.hidden_dim, cfg.masif_patches)
        self.masif_attn = MultiHeadCrossAttention(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.num_heads)
        self.esm_fc = nn.Sequential(nn.Linear(cfg.esm2_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))
        self.plif_fc = nn.Sequential(nn.Linear(cfg.plif_dim, 64), nn.ReLU(), nn.Linear(64, cfg.hidden_dim), ResidualMLP(cfg.hidden_dim))
        self.gated_fusion = GatedFusion(cfg.hidden_dim, num_modalities=5)
        self.fusion_dim = cfg.hidden_dim * 6
        self.classifier = nn.Sequential(nn.Linear(self.fusion_dim, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.projector = nn.Sequential(nn.Linear(self.fusion_dim, 256), nn.ReLU(), nn.Linear(256, cfg.projection_dim), nn.LayerNorm(cfg.projection_dim))
        self.ligand_ctx = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))
        self.protein_ctx = nn.Sequential(nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))

    def forward(self, ligand, esm, masif, plif):
        l_v = self.ligand_fc(ligand)
        m_p, m_g = self.masif_encoder(masif)
        i_v = self.masif_attn(l_v, m_p)
        e_v = self.esm_fc(esm)
        p_v = self.plif_fc(plif)
        f_v = self.gated_fusion([l_v, i_v, e_v, p_v, m_g])
        combined = torch.cat([f_v, l_v, i_v, e_v, p_v, m_g], dim=1)
        return self.classifier(combined)

# ==========================================
# 3. Model B Definition
# ==========================================
class ConfigB:
    ligand_dim = 2048; esm2_dim = 2560; hidden_dim = 512; dropout_prob = 0.0

class ResidualBlockB(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.net(x))

class PathB_Hunter_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ligand_encoder = nn.Sequential(nn.Linear(cfg.ligand_dim, cfg.hidden_dim), nn.BatchNorm1d(cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout_prob), ResidualBlockB(cfg.hidden_dim, cfg.dropout_prob))
        self.esm_encoder = nn.Sequential(nn.Linear(cfg.esm2_dim, cfg.hidden_dim), nn.BatchNorm1d(cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout_prob), ResidualBlockB(cfg.hidden_dim, cfg.dropout_prob))
        self.classifier = nn.Sequential(nn.Linear(cfg.hidden_dim * 2, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, ligand, esm):
        combined = torch.cat([self.ligand_encoder(ligand), self.esm_encoder(esm)], dim=1)
        return self.classifier(combined)

# ==========================================
# 4. Inference Dataset
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, df, plif_dict, morgan_dict, esm_root, masif_root):
        self.df = df.reset_index(drop=True)
        self.plif_dict = plif_dict
        self.morgan_dict = morgan_dict
        self.esm_root = esm_root
        self.masif_root = masif_root
        self.pid_col = 'protien_id' if 'protien_id' in df.columns else 'protein_id'

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row['Name'])
        pid_str = str(row[self.pid_col]).strip()

        # Features
        ligand = torch.tensor(self.morgan_dict.get(name, np.zeros(2048)), dtype=torch.float32)
        plif = torch.tensor(self.plif_dict.get(name, np.zeros(8)), dtype=torch.float32)
        
        # Load Protein (On-the-fly to save memory if dataset is huge, or cache if RAM allows)
        # Here we load on-the-fly with error handling
        esm_val = torch.zeros(2560)
        masif_val = torch.zeros((256, 80))
        
        try:
            esm_path = os.path.join(self.esm_root, f"{pid_str}.pt")
            if os.path.exists(esm_path):
                t = torch.load(esm_path, map_location='cpu')
                if t.dim() == 2: t = t.mean(dim=0)
                esm_val = t
        except: pass
        
        try:
            masif_path = os.path.join(self.masif_root, f"{pid_str}.pt")
            if os.path.exists(masif_path):
                t = torch.load(masif_path, map_location='cpu')
                if t.dim() == 2: t = t # Patch dim handled by shape check or pad
                elif t.dim() == 3: t = t.squeeze(0)
                if t.shape[0] == 256: masif_val = t
        except: pass

        return {
            'ligand': ligand,
            'esm': esm_val,
            'masif': masif_val,
            'plif': plif
        }

# ==========================================
# 5. Main Inference Logic
# ==========================================
def run_inference():
    print(f"==================================================")
    print(f"   PhysGater Ensemble Inference")
    print(f"   Paramters: Lambda_A={Config.LAMBDA_A}, Lambda_B={Config.LAMBDA_B}")
    print(f"   Threshold: {Config.FINAL_THRESHOLD}")
    print(f"==================================================\n")

    # 1. Load Data
    print("[1/3] Loading Benchmark Data...")
    df = pd.read_csv(Config.BENCHMARK_CSV)
    print(f"   Loaded {len(df)} molecules.")
    
    with open(Config.BENCHMARK_MORGAN_CACHE, 'rb') as f: morgan_d = pickle.load(f)
    with open(Config.BENCHMARK_PLIF_CACHE, 'rb') as f: plif_d = pickle.load(f)
    
    dataset = InferenceDataset(df, plif_d, morgan_d, Config.ESM2_ROOT, Config.MASIF_ROOT)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # 2. Ensemble Inference (10 Folds)
    ensemble_prob_a = np.zeros(len(df))
    ensemble_prob_b = np.zeros(len(df))
    
    print("\n[2/3] Running 10-Fold Ensemble Inference...")
    
    for fold in range(1, 11):
        print(f"   >>> Processing Fold {fold}/10...")
        
        # --- Path A ---
        path_a = os.path.join(Config.PATH_A_ROOT, f"fold_{fold}", "best_model.pth")
        if os.path.exists(path_a):
            model_a = MaSIFAttentionNet(ConfigA()).to(Config.DEVICE)
            model_a.load_state_dict(torch.load(path_a, map_location=Config.DEVICE))
            model_a.eval()
            
            preds_a = []
            with torch.no_grad():
                for batch in loader:
                    l = batch['ligand'].to(Config.DEVICE)
                    e = batch['esm'].to(Config.DEVICE)
                    m = batch['masif'].to(Config.DEVICE)
                    p = batch['plif'].to(Config.DEVICE)
                    logits = model_a(l, e, m, p)
                    preds_a.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            ensemble_prob_a += np.array(preds_a)
            del model_a
        else:
            print(f"      [Warning] Model A Fold {fold} missing, skipping.")

        # --- Path B ---
        path_b = os.path.join(Config.PATH_B_ROOT, f"fold_{fold}", "best_model.pth")
        if os.path.exists(path_b):
            model_b = PathB_Hunter_Net(ConfigB()).to(Config.DEVICE)
            model_b.load_state_dict(torch.load(path_b, map_location=Config.DEVICE))
            model_b.eval()
            
            preds_b = []
            with torch.no_grad():
                for batch in loader:
                    l = batch['ligand'].to(Config.DEVICE)
                    e = batch['esm'].to(Config.DEVICE)
                    logits = model_b(l, e)
                    preds_b.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            ensemble_prob_b += np.array(preds_b)
            del model_b
        else:
            print(f"      [Warning] Model B Fold {fold} missing, skipping.")
            
        torch.cuda.empty_cache()

    # 3. Average & Fuse
    print("\n[3/3] Calculating Final Scores...")
    # English comment: see script logic.
    avg_prob_a = ensemble_prob_a / 10.0
    avg_prob_b = ensemble_prob_b / 10.0
    
    # English comment: see script logic.
    # English comment: see script logic.
    fused_score = np.exp(
        Config.LAMBDA_A * np.log(avg_prob_a + 1e-6) + 
        Config.LAMBDA_B * np.log(avg_prob_b + 1e-6)
    )
    
    final_pred = (fused_score > Config.FINAL_THRESHOLD).astype(int)
    
    # 4. Save
    result_df = df.copy()
    result_df['Score_Reviewer_A'] = avg_prob_a
    result_df['Score_Hunter_B'] = avg_prob_b
    result_df['PhysGater_Score'] = fused_score
    result_df['PhysGater_Pred'] = final_pred
    
    result_df.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✅ Inference Complete! Results saved to: {Config.OUTPUT_FILE}")
    print(f"   Total Predicted Actives: {final_pred.sum()} / {len(df)}")
    
    # English comment: see script logic.
    plt.figure(figsize=(8, 6))
    sns.histplot(fused_score, bins=50, kde=True)
    plt.axvline(x=Config.FINAL_THRESHOLD, color='red', linestyle='--', label='Cutoff')
    plt.title('Benchmark Score Distribution')
    plt.xlabel('PhysGater Geometric Score')
    plt.savefig('benchmark_score_dist.png')
    print("   Distribution plot saved to 'benchmark_score_dist.png'")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    run_inference()