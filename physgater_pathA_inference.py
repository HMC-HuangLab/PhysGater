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
    weights_root = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3"  # English comment removed for consistency.
    
    # English comment: see script logic.
    esm2_root = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/DEKOIS_ESM2_FEAT/"
    masif_root = "/mnt/data/multitaskfp/DEKOIS_2.0/DEKOIS_targets/dekois_256_pro_feat/"
    plif_root = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/DEKOIS2_PLIF_Flat/"  # English comment removed for consistency.

    # English comment: see script logic.
    plif_cache = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/dekois_plif_cache.pkl"
    morgan_cache = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/dekois_morgan_cache.pkl"

    # English comment: see script logic.
    output_csv_path = "/mnt/data/multitaskfp/DEKOIS_2.0/merged_results/pathA_inference_predictions.csv"

    # English comment: see script logic.
    ligand_dim, esm2_dim, masif_dim = 2048, 2560, 80
    masif_patches, plif_dim, hidden_dim = 256, 8, 256
    projection_dim, num_heads = 128, 4
    bit_dropout_prob = 0.05
    
    # English comment: see script logic.
    batch_size = 64
    model_prob_threshold = 0.5 
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# English comment: see script logic.
# ==========================================
class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, embed_dim, num_heads):
        super().__init__()
        self.num_heads, self.embed_dim = num_heads, embed_dim
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj, self.v_proj = nn.Linear(kv_dim, embed_dim), nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
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
        self.groups = groups
        self.global_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU())
    def forward(self, masif):
        x = self.blocks(self.input_proj(masif))
        mean_pool, max_pool = x.mean(dim=1), x.amax(dim=1)
        chunks = torch.chunk(x, min(self.groups, x.size(1)), dim=1)
        chunk_means = torch.stack([c.mean(dim=1) for c in chunks], dim=1).mean(dim=1)
        global_vec = self.global_fusion(torch.cat([mean_pool, max_pool, chunk_means], dim=-1))
        return x, global_vec

class GatedFusion(nn.Module):
    def __init__(self, feature_dim, num_modalities):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(feature_dim * num_modalities, feature_dim), nn.ReLU(), nn.Linear(feature_dim, num_modalities))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        weights = self.softmax(self.gate(stacked.flatten(1))).unsqueeze(-1)
        return (stacked * weights).sum(dim=1), weights.squeeze(-1)

class MaSIFAttentionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ligand_fc = nn.Sequential(nn.Linear(cfg.ligand_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.bit_dropout_prob), ResidualMLP(cfg.hidden_dim), ResidualMLP(cfg.hidden_dim))
        self.masif_encoder = MasifPatchEncoder(cfg.masif_dim, cfg.hidden_dim, cfg.masif_patches)
        self.masif_attn = MultiHeadCrossAttention(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.num_heads)
        self.esm_fc = nn.Sequential(nn.Linear(cfg.esm2_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))
        self.plif_fc = nn.Sequential(nn.Linear(cfg.plif_dim, 64), nn.ReLU(), nn.Linear(64, cfg.hidden_dim), ResidualMLP(cfg.hidden_dim))
        self.gated_fusion = GatedFusion(cfg.hidden_dim, 5)
        self.classifier = nn.Sequential(nn.Linear(cfg.hidden_dim * 6, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        # English comment: see script logic.
        self.projector = nn.Sequential(nn.Linear(cfg.hidden_dim * 6, 256), nn.ReLU(), nn.Linear(256, cfg.projection_dim), nn.LayerNorm(cfg.projection_dim))
        self.ligand_ctx = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))
        self.protein_ctx = nn.Sequential(nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))

    def forward(self, ligand, esm, masif, plif):
        l_v = self.ligand_fc(ligand)
        m_p, m_g = self.masif_encoder(masif)
        i_v = self.masif_attn(l_v, m_p)
        e_v = self.esm_fc(esm)
        p_v = self.plif_fc(plif)
        f_v, _ = self.gated_fusion([l_v, i_v, e_v, p_v, m_g])
        return self.classifier(torch.cat([f_v, l_v, i_v, e_v, p_v, m_g], dim=1))

# ==========================================
# English comment: see script logic.
# ==========================================
class PLIFProcessor:
    @staticmethod
    def process(plif_root, cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f: return pickle.load(f)
        plif_dict = {}
        interaction_types = ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", "Anionic", "Cationic", "CationPi", "PiCation"]
        csv_files = glob.glob(os.path.join(plif_root, "*_plif_features.csv"))
        for file_path in tqdm(csv_files, desc="Parsing PLIF"):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                feat = []
                for itype in interaction_types:
                    cols = [c for c in df.columns if itype in c]
                    feat.append(row[cols].sum() if cols else 0)
                plif_dict[str(row['Name'])] = np.array(feat, dtype=np.float32)
        with open(cache_path, 'wb') as f: pickle.dump(plif_dict, f)
        return plif_dict

class MorganProcessor:
    @staticmethod
    def process(csv_path, cache_path, n_bits=2048):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f: return pickle.load(f)
        df = pd.read_csv(csv_path)
        smi_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
        morgan_dict = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Gen Morgan"):
            try:
                mol = Chem.MolFromSmiles(row[smi_col])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                morgan_dict[str(row['Name'])] = np.array(fp, dtype=np.float32)
            except:
                morgan_dict[str(row['Name'])] = np.zeros((n_bits,), dtype=np.float32)
        with open(cache_path, 'wb') as f: pickle.dump(morgan_dict, f)
        return morgan_dict

class MaSIFInferenceDataset(Dataset):
    def __init__(self, df, plif_dict, morgan_dict, config):
        self.df, self.plif_dict, self.morgan_dict, self.cfg = df.reset_index(drop=True), plif_dict, morgan_dict, config
        self.pid_col = 'protien_id' if 'protien_id' in df.columns else 'protein_id'
        self.p_cache = {}
        # English comment: see script logic.
        for pid in tqdm(self.df[self.pid_col].unique(), desc="Loading Protein Features", leave=False):
            pid_s = str(pid).strip()
            e_p, m_p = os.path.join(config.esm2_root, f"{pid_s}.pt"), os.path.join(config.masif_root, f"{pid_s}.pt")
            
            # English comment: see script logic.
            if os.path.exists(e_p):
                ev = torch.load(e_p, map_location='cpu')
                if ev.dim() > 1: ev = ev.mean(0)  # English comment removed for consistency.
            else: ev = torch.zeros(config.esm2_dim)
            
            if os.path.exists(m_p):
                mv = torch.load(m_p, map_location='cpu')
                if mv.dim() == 3: mv = mv.squeeze(0)  # English comment removed for consistency.
            else: mv = torch.zeros((config.masif_patches, config.masif_dim))
            
            self.p_cache[pid_s] = {'esm': ev, 'masif': mv}

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name, pid = str(row['Name']), str(row[self.pid_col]).strip()
        return {
            'ligand': torch.tensor(self.morgan_dict.get(name, np.zeros(self.cfg.ligand_dim)), dtype=torch.float32),
            'esm': self.p_cache[pid]['esm'], 'masif': self.p_cache[pid]['masif'],
            'plif': torch.tensor(self.plif_dict.get(name, np.zeros(self.cfg.plif_dim)), dtype=torch.float32)
        }

# ==========================================
# English comment: see script logic.
# ==========================================
if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    print("[1/3] Loading Features...")
    df = pd.read_csv(Config.test_csv_path)
    plif_data = PLIFProcessor.process(Config.plif_root, Config.plif_cache)
    morgan_data = MorganProcessor.process(Config.test_csv_path, Config.morgan_cache, Config.ligand_dim)
    
    print("[2/3] Loading 10-Fold Ensemble Models...")
    models = []
    for i in range(10):
        p = os.path.join(Config.weights_root, f"fold_{i+1}", "best_model.pth")
        if os.path.exists(p):
            m = MaSIFAttentionNet(Config).to(Config.device)
            m.load_state_dict(torch.load(p, map_location=Config.device), strict=False)
            m.eval()
            models.append(m)
    
    if not models: raise RuntimeError("未加载任何模型权重。")

    results = []
    loader = DataLoader(MaSIFInferenceDataset(df, plif_data, morgan_data, Config), batch_size=Config.batch_size, shuffle=False)

    print(f"[3/3] Inference on {len(df)} samples...")
    # English comment: see script logic.
    all_fold_probs = [[] for _ in range(len(models))]
    
    with torch.no_grad():
        for batch in tqdm(loader):
            l, e, m, pl = batch['ligand'].to(Config.device), batch['esm'].to(Config.device), batch['masif'].to(Config.device), batch['plif'].to(Config.device)
            for i, model in enumerate(models):
                # English comment: see script logic.
                out = torch.sigmoid(model(l, e, m, pl)).cpu().numpy().flatten()
                all_fold_probs[i].extend(out)
    
    matrix = np.array(all_fold_probs) # [10, N]
    means = matrix.mean(0)
    # 95% CI: Mean +/- 1.96 * (Std / sqrt(10))
    stds = matrix.std(0, ddof=1)
    ci_half = 1.96 * (stds / math.sqrt(len(models)))

    for i in range(len(df)):
        results.append({
            'Name': df.loc[i, 'Name'],
            'Target_Name': df.loc[i, 'Target_Name'] if 'Target_Name' in df.columns else 'Unknown',
            'Prediction_Label': 1 if means[i] >= Config.model_prob_threshold else 0,
            'Confidence_Interval': f"[{means[i]-ci_half[i]:.4f}, {means[i]+ci_half[i]:.4f}]"
        })

    pd.DataFrame(results).to_csv(Config.output_csv_path, index=False)
    print(f"\n[Done] PathA 推理完成！结果已保存至: {Config.output_csv_path}")