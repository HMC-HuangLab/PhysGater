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
import glob
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
# Added specific metrics
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import shutil
import warnings
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.model_selection import GroupKFold
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

# Check RDKit installation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit is not installed. Please run: pip install rdkit")

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # English comment: see script logic.
    csv_path = "/mnt/data/fpdetec_V2/final_with_thresholds/new_train_data_relabelled.csv"
    esm2_root = "/mnt/data/fpdetec_V2/new_esm2_feat"
    masif_root = "/mnt/data/fpdetec_V2/pocket_256_masif_feat"
    plif_root = "/mnt/data/fpdetec_V2/Dataset_PLIF_Flat"

    plif_cache = "/mnt/data/fpdetec_V2/plif_processed_cache.pkl"
    morgan_cache = "/mnt/data/fpdetec_V2/morgan_2048_cache.pkl"

    # English comment: see script logic.
    output_dir = "/mnt/data/fpdetec_V2/PathA_FP_Suppression_V3" 

    # Dimensions
    ligand_dim = 2048
    esm2_dim = 2560
    masif_dim = 80
    masif_patches = 256
    plif_dim = 8

    hidden_dim = 256
    projection_dim = 128
    num_heads = 4

    # Training Hyperparameters
    batch_size = 64
    lr = 5e-5
    epochs = 50

    eval_threshold = 0.5
    
    # English comment: see script logic.
    margin = 0.2            # Ranking Loss Margin
    contrastive_temp = 0.07
    contrastive_weight = 0.2 
          
    active_weight=3.0  # English comment removed for consistency.
    fp_weight=2.0  # English comment removed for consistency.
    ranking_weight=2.0  # English comment removed for consistency.
    easy_neg_weight = 0.5  # English comment removed for consistency.

    use_focal_loss = True
    # English comment: see script logic.
    focal_gamma = 2.0

    # English comment: see script logic.
    patience = 7  # English comment removed for consistency.
    min_delta = 0.0001  # English comment removed for consistency.

    ema_decay = 0.999
    use_amp = True

    # Modality Dropout Probabilities
    masif_dropout_prob = 0.15
    plif_dropout_prob = 0.1
    patch_dropout_prob = 0.2
    bit_dropout_prob = 0.05

    ligand_dropout_prob = 0.6
    gate_entropy_weight = 0.05

    seed = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(Config.seed)

# ==========================================
# English comment: see script logic.
# ==========================================
class PLIFProcessor:
    @staticmethod
    def process(plif_root, cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"[Info] Processing PLIFs from CSVs...")
        plif_dict = {}
        interaction_types = ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
                             "Anionic", "Cationic", "CationPi", "PiCation"]

        csv_files = glob.glob(os.path.join(plif_root, "*_plif_features.csv"))
        for file_path in tqdm(csv_files, desc="Parsing PLIF"):
            try:
                df = pd.read_csv(file_path)
                names = df['Name'].values
                feats = np.zeros((len(df), len(interaction_types)), dtype=np.float32)
                for i, itype in enumerate(interaction_types):
                    cols = [c for c in df.columns if itype in c]
                    if cols:
                        feats[:, i] = df[cols].sum(axis=1).values
                for idx, name in enumerate(names):
                    plif_dict[name] = feats[idx]
            except Exception:
                pass

        with open(cache_path, 'wb') as f:
            pickle.dump(plif_dict, f)
        return plif_dict

class MorganProcessor:
    @staticmethod
    def process(csv_path, cache_path, n_bits=2048):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"[Info] Generating Morgan Fingerprints...")
        df = pd.read_csv(csv_path)
        smi_col = 'smiles' if 'smiles' in df.columns else 'SMILES'

        morgan_dict = {}
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Gen Morgan"):
            name = str(row['Name'])
            smi = row[smi_col]
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                    morgan_dict[name] = np.array(fp, dtype=np.float32)
                else:
                    morgan_dict[name] = np.zeros((n_bits,), dtype=np.float32)
            except Exception:
                morgan_dict[name] = np.zeros((n_bits,), dtype=np.float32)

        with open(cache_path, 'wb') as f:
            pickle.dump(morgan_dict, f)
        return morgan_dict

# ==========================================
# English comment: see script logic.
# ==========================================
class MaSIFDataset(Dataset):
    def __init__(self, df, plif_dict, morgan_dict, config):
        self.df = df.reset_index(drop=True)
        self.plif_dict = plif_dict
        self.morgan_dict = morgan_dict
        self.cfg = config
        self.pid_col = 'protien_id' if 'protien_id' in df.columns else 'protein_id'

        print("[Info] Preloading Proteins (Keeping MaSIF Patches)...")
        self.protein_cache = {}
        unique_pids = self.df[self.pid_col].unique()

        for pid in tqdm(unique_pids, desc="Loading Protein Features"):
            pid_str = str(pid).strip()

            # ESM2 Global pooling
            esm_path = os.path.join(config.esm2_root, f"{pid_str}.pt")
            esm_val = torch.zeros(config.esm2_dim)
            if os.path.exists(esm_path):
                try:
                    t = torch.load(esm_path, map_location='cpu')
                    if t.dim() == 2: t = t.mean(dim=0)
                    esm_val = t
                except Exception: pass

            # MaSIF patches
            masif_path = os.path.join(config.masif_root, f"{pid_str}.pt")
            masif_val = torch.zeros((config.masif_patches, config.masif_dim))
            if os.path.exists(masif_path):
                try:
                    t = torch.load(masif_path, map_location='cpu')
                    if t.dim() == 2 and t.shape[0] == config.masif_patches:
                        masif_val = t
                    elif t.dim() == 3:
                        masif_val = t.squeeze(0)
                    if masif_val.shape[0] != config.masif_patches:
                        masif_val = torch.zeros((config.masif_patches, config.masif_dim))
                except Exception: pass

            self.protein_cache[pid_str] = {'esm': esm_val, 'masif': masif_val}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row['Name'])
        pid_str = str(row[self.pid_col]).strip()

        ligand = torch.tensor(self.morgan_dict.get(name, np.zeros(self.cfg.ligand_dim)), dtype=torch.float32)
        p_data = self.protein_cache.get(pid_str, {
            'esm': torch.zeros(self.cfg.esm2_dim),
            'masif': torch.zeros((self.cfg.masif_patches, self.cfg.masif_dim))
        })
        plif = torch.tensor(self.plif_dict.get(name, np.zeros(self.cfg.plif_dim)), dtype=torch.float32)

        return {
            'ligand': ligand,
            'esm': p_data['esm'],
            'masif': p_data['masif'],
            'plif': plif,
            'label_bio': torch.tensor(float(row['Ground_Truth']), dtype=torch.float32),
            'label_cls': torch.tensor(int(row['Label_Idx']), dtype=torch.long)
        }

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
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim: raise ValueError("embed_dim divisible by num_heads")
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
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
        x = self.input_proj(masif)
        x = self.blocks(x)
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
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
        weights = self.softmax(self.gate(stacked.flatten(start_dim=1))).unsqueeze(-1)
        return (stacked * weights).sum(dim=1), weights.squeeze(-1)

class MaSIFAttentionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ligand_fc = nn.Sequential(
            nn.Linear(cfg.ligand_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(),
            nn.Dropout(cfg.bit_dropout_prob), ResidualMLP(cfg.hidden_dim), ResidualMLP(cfg.hidden_dim)
        )
        self.masif_encoder = MasifPatchEncoder(cfg.masif_dim, cfg.hidden_dim, cfg.masif_patches)
        self.masif_attn = MultiHeadCrossAttention(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.num_heads)
        self.esm_fc = nn.Sequential(nn.Linear(cfg.esm2_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))
        self.plif_fc = nn.Sequential(nn.Linear(cfg.plif_dim, 64), nn.ReLU(), nn.Linear(64, cfg.hidden_dim), ResidualMLP(cfg.hidden_dim))
        self.gated_fusion = GatedFusion(cfg.hidden_dim, num_modalities=5)
        self.fusion_dim = cfg.hidden_dim * 6
        
        # Classifier (NO Sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.projector = nn.Sequential(
            nn.Linear(self.fusion_dim, 256), nn.ReLU(), nn.Linear(256, cfg.projection_dim), nn.LayerNorm(cfg.projection_dim)
        )
        self.ligand_ctx = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))
        self.protein_ctx = nn.Sequential(nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim), nn.ReLU(), ResidualMLP(cfg.hidden_dim))

    def forward(self, ligand, esm, masif, plif):
        ligand_vec = self.ligand_fc(ligand)
        masif_patches, masif_global = self.masif_encoder(masif)
        inter_vec = self.masif_attn(ligand_vec, masif_patches)
        esm_vec = self.esm_fc(esm)
        plif_vec = self.plif_fc(plif)

        fused_vec, gate_weights = self.gated_fusion([ligand_vec, inter_vec, esm_vec, plif_vec, masif_global])
        combined = torch.cat([fused_vec, ligand_vec, inter_vec, esm_vec, plif_vec, masif_global], dim=1)

        logits = self.classifier(combined)
        projection = self.projector(combined)
        ligand_ctx = self.ligand_ctx(ligand_vec)
        protein_ctx = self.protein_ctx(torch.cat([inter_vec, esm_vec, masif_global], dim=1))

        return logits, projection, ligand_ctx, protein_ctx, gate_weights

# ==========================================
# English comment: see script logic.
# ==========================================
class FocalLoss(nn.Module):
    """
    实现 Focal Loss 以解决类别不平衡及难易样本权重问题
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # English comment: see script logic.
        p = torch.sigmoid(inputs)
        # English comment: see script logic.
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # English comment: see script logic.
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class HybridRankingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.margin = cfg.margin
        self.ranking_weight = cfg.ranking_weight
        # English comment: see script logic.
        self.bce_none = nn.BCEWithLogitsLoss(reduction='none')
        
        self.active_w = cfg.active_weight
        self.fp_w = cfg.fp_weight
        self.easy_neg_w = getattr(cfg, 'easy_neg_weight', 0.5)
        self.gamma = getattr(cfg, 'focal_gamma', 2.0)

    def forward(self, logits, labels_bio, labels_cls):
        """
        labels_cls: 0=EasyNeg, 1=Active, 2=HardNeg (FP form Path B)
        """
        # --- 1. Weighted Focal Loss ---
        # English comment: see script logic.
        bce_loss = self.bce_none(logits.view(-1), labels_bio)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        weights = torch.ones_like(logits.view(-1))
        weights[labels_cls == 0] = self.easy_neg_w
        weights[labels_cls == 1] = self.active_w
        weights[labels_cls == 2] = self.fp_w
        
        cls_loss = (focal_loss * weights).mean()
        
        # --- 2. Pairwise Ranking Loss ---
        pos_mask = (labels_cls == 1)
        hard_neg_mask = (labels_cls == 2)
        
        if pos_mask.sum() > 0 and hard_neg_mask.sum() > 0:
            # English comment: see script logic.
            pos_scores = logits[pos_mask].view(-1) 
            neg_scores = logits[hard_neg_mask].view(-1)
            
            n_pos = pos_scores.size(0)
            n_neg = neg_scores.size(0)
            
            # English comment: see script logic.
            # pos: [n_pos] -> [n_pos, 1] -> [n_pos, n_neg]
            pos_exp = pos_scores.unsqueeze(1).expand(n_pos, n_neg).reshape(-1)
            
            # neg: [n_neg] -> [1, n_neg] -> [n_pos, n_neg]
            neg_exp = neg_scores.unsqueeze(0).expand(n_pos, n_neg).reshape(-1)
            
            target = torch.ones_like(pos_exp)
            rank_loss = F.margin_ranking_loss(pos_exp, neg_exp, target, margin=self.margin)
        else:
            rank_loss = torch.tensor(0.0, device=logits.device)
            
        return cls_loss + self.ranking_weight * rank_loss, rank_loss
    
# ==========================================
# English comment: see script logic.
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        # self.criterion = HybridRankingLoss(margin=config.margin, ranking_weight=config.ranking_weight).to(device)
        self.criterion = HybridRankingLoss(config).to(device)
        self.scaler = GradScaler(enabled=config.use_amp and torch.cuda.is_available())
        self.ema_model = None
        if 0 < config.ema_decay < 1.0:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(device)
            for param in self.ema_model.parameters(): param.requires_grad = False
        if os.path.exists(config.output_dir): shutil.rmtree(config.output_dir)
        os.makedirs(config.output_dir)
        self.output_dir = config.output_dir
        
        # English comment: see script logic.
        self.patience = getattr(config, 'patience', 10)
        self.min_delta = getattr(config, 'min_delta', 0.0)
        self.counter = 0
        self.early_stop = False
        self.best_score = -float('inf')  # English comment removed for consistency.

        # English comment: see script logic.
        self.best_f1 = 0.0
        self.best_auc = 0.0  # English comment removed for consistency.
        self.history = {'train_loss': [], 'val_auc': [], 'val_f1': [], 'val_fp_rej': []}
        # self.history = {'train_loss': [], 'val_auc': [], 'val_f1':  [], 'val_fp_rej': []}

    def update_ema(self):
        if self.ema_model is None: return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.cfg.ema_decay).add_(param.data, alpha=1 - self.cfg.ema_decay)
            for ema_buf, buf in zip(self.ema_model.buffers(), self.model.buffers()): ema_buf.copy_(buf)

    def get_eval_model(self): return self.ema_model if self.ema_model is not None else self.model

    def compute_contrastive_loss(self, ligand_emb, protein_emb):
        if ligand_emb.size(0) < 2: 
            return torch.tensor(0.0, device=self.device)
            
        # English comment: see script logic.
        lig = F.normalize(ligand_emb, dim=1, eps=1e-6)
        prot = F.normalize(protein_emb, dim=1, eps=1e-6)
        
        # English comment: see script logic.
        logits = torch.matmul(lig, prot.T) / self.cfg.contrastive_temp
        logits = torch.clamp(logits, max=100.0) 
        
        labels = torch.arange(lig.size(0), device=self.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def _apply_modality_dropout(self, tensor, prob):
        if prob <= 0.0: return tensor
        mask_shape = [tensor.size(0)] + [1] * (tensor.dim() - 1)
        mask = (torch.rand(mask_shape, device=self.device) > prob).float()
        return tensor * mask

    def _apply_patch_dropout(self, masif):
        if self.cfg.patch_dropout_prob <= 0.0: return masif
        patch_mask = (torch.rand(masif.size(0), masif.size(1), 1, device=self.device) > self.cfg.patch_dropout_prob).float()
        return masif * patch_mask

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs} [Train]")
        
        # English comment: see script logic.
        nan_counter = 0 
        
        for batch in loop:
            ligand = batch['ligand'].to(self.device)
            esm = batch['esm'].to(self.device)
            masif = batch['masif'].to(self.device)
            plif = batch['plif'].to(self.device)
            labels_bio = batch['label_bio'].to(self.device)
            labels_cls = batch['label_cls'].to(self.device)

            # English comment: see script logic.
            if self.cfg.ligand_dropout_prob > 0:
                mask = (torch.rand(ligand.size(0), 1, device=self.device) > self.cfg.ligand_dropout_prob).float()
                ligand = ligand * mask 
            
            if getattr(self.cfg, 'masif_dropout_prob', 0) > 0:
                 mask_m = (torch.rand(masif.size(0), 1, 1, device=self.device) > self.cfg.masif_dropout_prob).float()
                 masif = masif * mask_m
            
            if getattr(self.cfg, 'patch_dropout_prob', 0) > 0:
                 mask_p = (torch.rand(masif.size(0), masif.size(1), 1, device=self.device) > self.cfg.patch_dropout_prob).float()
                 masif = masif * mask_p

            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=self.cfg.use_amp):
                # Forward
                logits, projection, lig_ctx, prot_ctx, gate_weights = self.model(ligand, esm, masif, plif)
                
                # English comment: see script logic.
                loss, rank_loss = self.criterion(logits, labels_bio, labels_cls)
                
                # English comment: see script logic.
                if self.cfg.contrastive_weight > 0 and lig_ctx.size(0) > 1:
                    contrastive = self.compute_contrastive_loss(lig_ctx, prot_ctx)
                else:
                    contrastive = torch.tensor(0.0, device=self.device)

                # English comment: see script logic.
                # English comment: see script logic.
                gate_safe = torch.clamp(gate_weights, min=1e-6, max=1.0)
                entropy = -torch.sum(gate_safe * torch.log(gate_safe), dim=1).mean()
                reg_loss = -entropy 
                
                # English comment: see script logic.
                final_loss = loss + \
                             (self.cfg.contrastive_weight * contrastive) + \
                             (self.cfg.gate_entropy_weight * reg_loss)

            # English comment: see script logic.
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                nan_counter += 1
                print(f"[Warning] NaN detected (Count: {nan_counter}). Loss details -> Main: {loss.item()}, Ctr: {contrastive.item()}, Reg: {reg_loss.item()}")
                
                # English comment: see script logic.
                if nan_counter > 5:
                    print("Error: Model weights corrupted (Continuous NaNs). Stopping training.")
                    raise ValueError("Model Collapse")
                
                self.optimizer.zero_grad()
                continue
            else:
                nan_counter = 0  # English comment removed for consistency.

            # Backward
            self.scaler.scale(final_loss).backward()
            
            # English comment: see script logic.
            self.scaler.unscale_(self.optimizer)
            # English comment: see script logic.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_ema()

            total_loss += final_loss.item()
            
            # English comment: see script logic.
            avg_gates = gate_weights.mean(0).detach().cpu().numpy().round(2)
            loop.set_postfix(
                loss=f"{final_loss.item():.2f}", 
                ctr=f"{contrastive.item():.2f}",
                gate=str(avg_gates)
            )

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        eval_model = self.get_eval_model()
        eval_model.eval()
        all_preds, all_labels, all_cls_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[Valid]"):
                ligand = batch['ligand'].to(self.device)
                esm = batch['esm'].to(self.device)
                masif = batch['masif'].to(self.device)
                plif = batch['plif'].to(self.device)
                logits, _, _, _, _ = eval_model(ligand, esm, masif, plif)
                pred = torch.sigmoid(logits)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch['label_bio'].numpy())
                all_cls_labels.extend(batch['label_cls'].numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)
        
        # Binary predictions with threshold 0.5
        all_pred_binary = (all_preds > Config.eval_threshold).astype(int)

        # 1. Calculate Metrics
        auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        f1 = f1_score(all_labels, all_pred_binary, zero_division=0)
        recall = recall_score(all_labels, all_pred_binary, zero_division=0)
        precision = precision_score(all_labels, all_pred_binary, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_pred_binary)

        all_cls_labels = np.array(all_cls_labels)
        fp_indices = (all_cls_labels == 2)
        fp_rejection = (all_preds[fp_indices] < Config.eval_threshold).mean() if fp_indices.sum() > 0 else 0.0

        print(f"Epoch {epoch+1}: AUC={auc_score:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Prec={precision:.4f}, MCC={mcc:.4f}, FP_Rej={fp_rejection:.4f}")
        
        # English comment: see script logic.
        current_score = 0.4 * f1 + 0.6 * precision        
                # English comment: see script logic.
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # English comment removed for consistency.
            
            # English comment: see script logic.
            torch.save(eval_model.state_dict(), f"{self.cfg.output_dir}/best_model.pth")
            print(f"*** New Best Model Saved (Score: {current_score:.4f} | Prec: {precision:.4f}, F1: {f1:.4f}) ***")
        else:
            self.counter += 1
            print(f"[EarlyStopping] Counter: {self.counter} / {self.patience} (Best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"*** Early Stopping Triggered at Epoch {epoch+1} ***")

        # English comment: see script logic.
        self.scheduler.step(f1)
        self.history['val_auc'].append(auc_score)
        self.history['val_f1'].append(f1)
        self.history['val_fp_rej'].append(fp_rejection)
        return f1, auc_score, fp_rejection
    
    def save_history(self):
        """保存训练曲线原始数据"""
        with open(os.path.join(self.output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)

def plot_results(trainer, output_dir):
    epochs = range(1, len(trainer.history['train_loss']) + 1)
    plt.style.use('default') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, trainer.history['train_loss'], 'b-o', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # English comment: see script logic.
    ax2.plot(epochs, trainer.history['val_auc'], 'r--s', alpha=0.6, label='Val AUC')
    ax2.plot(epochs, trainer.history['val_f1'], 'g-o', alpha=0.8, label='Val F1')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=800)
    plt.close()

# ==========================================
# English comment: see script logic.
# ==========================================
def plot_cross_validation_roc(cv_results, output_dir):
    """
    绘制 N 折交叉验证的平均 ROC 曲线 (带标准差阴影)
    """
    plt.figure(figsize=(8, 8), dpi=600)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # English comment: see script logic.
    for i, res in enumerate(cv_results):
        # English comment: see script logic.
        interp_tpr = np.interp(mean_fpr, res['fpr'], res['tpr'])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(res['auc'])
        plt.plot(res['fpr'], res['tpr'], lw=1, alpha=0.3, 
                 label=f'Fold {i+1} (AUC = {res["auc"]:.3f})')
    
    # English comment: see script logic.
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})', lw=2, alpha=0.8)
    
    # English comment: see script logic.
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Validation ROC Curve')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(output_dir, "cv_mean_roc_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Info] Mean ROC curve saved to {save_path}")

def plot_cv_metrics_summary(cv_metrics_dict, output_dir):
    """
    绘制十折交叉验证的所有指标（AUC, F1, Recall, Precision, MCC）的箱线图
    """
    df_metrics = pd.DataFrame(cv_metrics_dict)
    
    plt.figure(figsize=(10, 6), dpi=600)
    sns.boxplot(data=df_metrics, palette="Set2")
    sns.stripplot(data=df_metrics, color=".25", size=4, alpha=0.6)
    
    plt.title(f'{len(df_metrics)}-Fold Cross-Validation Metrics Summary')
    plt.ylabel('Score')
    plt.grid(True, axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, "cv_metrics_boxplot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Info] CV Metrics boxplot saved to {save_path}")

def plot_cross_validation_pr(cv_pr_data, output_dir):
    """新增：平均 Precision-Recall 曲线"""
    plt.figure(figsize=(8, 8), dpi=600)
    base_recall = np.linspace(0, 1, 100)
    precisions = []
    
    for res in cv_pr_data:
        # English comment: see script logic.
        # English comment: see script logic.
        interp_prec = np.interp(base_recall, res['recall'][::-1], res['precision'][::-1])
        precisions.append(interp_prec)
        plt.plot(res['recall'], res['precision'], lw=1, alpha=0.3)
    
    mean_prec = np.mean(precisions, axis=0)
    plt.plot(base_recall, mean_prec, color='g', label='Mean PR Curve', lw=2)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Cross-Validation PR Curve')
    plt.legend(); plt.savefig(os.path.join(output_dir, "cv_mean_pr_curve.png"), dpi=600); plt.close()



def plot_cv_radar_chart(cv_metrics, output_dir):
    """新增：雷达图展示各项指标的平均水平"""
    labels = list(cv_metrics.keys())
    stats = [np.mean(cv_metrics[l]) for l in labels]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    stats = stats + stats[:1]; angles = angles + angles[:1]  # English comment removed for consistency.
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='red', alpha=0.25)
    ax.plot(angles, stats, color='red', linewidth=2)
    ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    plt.title('Average Performance Radar Chart')
    plt.savefig(os.path.join(output_dir, "cv_metrics_radar.png"), dpi=600); plt.close()

class AdvancedVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # English comment: see script logic.
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_pr_curve(self, y_true, y_scores):
        """1. 绘制 Precision-Recall 曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(recall, precision, color='#2ca02c', lw=2, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'), dpi=600)
        plt.close()

    def plot_score_distribution(self, y_true, y_scores):
        """2. 绘制正负样本的预测分数分布"""
        df = pd.DataFrame({'Score': y_scores, 'Label': y_true})
        df['Label'] = df['Label'].map({0: 'Inactive', 1: 'Active'})

        plt.figure(figsize=(8, 6), dpi=600)
        sns.histplot(data=df, x='Score', hue='Label', bins=30, kde=True, element="step", stat="density", common_norm=False, palette=['#1f77b4', '#d62728'])
        plt.axvline(x=Config.eval_threshold, color='red', linestyle='--', label=f'Threshold={Config.eval_threshold}')
        plt.title(f'Prediction Score Distribution (Threshold={Config.eval_threshold})')
        plt.xlabel('Predicted Probability')
        plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'), dpi=600)
        plt.close()

    def plot_calibration_curve(self, y_true, y_scores):
        """3. 绘制校准曲线 (Reliability Diagram)"""
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)

        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'calibration_curve.png'), dpi=600)
        plt.close()

    def plot_modality_weights(self, gate_weights_list):
        """4. 绘制多模态融合的权重贡献 (Boxplot)"""
        # gate_weights shape: [N, 5] -> [Ligand, Interaction, ESM, PLIF, MaSIF_Global]
        # English comment: see script logic.
        if len(gate_weights_list) == 0:
            return
            
        weights = np.concatenate(gate_weights_list, axis=0)
        modality_names = ['Ligand', 'Attn_Inter', 'ESM2', 'PLIF', 'MaSIF_Global']
        
        df_weights = pd.DataFrame(weights, columns=modality_names)
        df_melt = df_weights.melt(var_name='Modality', value_name='Gate Weight')

        plt.figure(figsize=(10, 6), dpi=600)
        sns.boxplot(x='Modality', y='Gate Weight', data=df_melt, palette="Set3", showfliers=False)
        sns.stripplot(x='Modality', y='Gate Weight', data=df_melt, color=".25", alpha=0.1, size=2)
        plt.title('Modality Importance (Gated Fusion Weights)')
        plt.ylabel('Attention Weight')
        plt.savefig(os.path.join(self.output_dir, 'modality_importance.png'), dpi=600)
        plt.close()

    def plot_tsne(self, embeddings, y_true, max_points=2000):
        """5. t-SNE 降维可视化 (采样以加快速度)"""
        if len(embeddings) == 0:
            return
            
        # English comment: see script logic.
        n_samples = len(embeddings)
        if n_samples > max_points:
            indices = np.random.choice(n_samples, max_points, replace=False)
            embeddings = embeddings[indices]
            y_true = y_true[indices]
        
        print("[Info] Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 8), dpi=600)
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, cmap='coolwarm', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Label (0=Inactive, 1=Active)')
        plt.title('t-SNE of Model Embeddings')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(os.path.join(self.output_dir, 'tsne_embedding.png'), dpi=600)
        plt.close()
    
    def plot_enrichment_factor_curve(self, y_true, y_scores):
        """
        2. 富集因子 (EF) 随筛选比例变化的曲线图
        展示模型在 Top 1%, 5%, 10% 的捕获效率
        """
        ratios = np.linspace(0.001, 0.2, 100)  # English comment removed for consistency.
        ef_values = []
        
        n_pos_total = np.sum(y_true)
        n_total = len(y_true)
        random_baseline = n_pos_total / n_total

        # English comment: see script logic.
        indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[indices]

        for r in ratios:
            n_top = max(1, int(n_total * r))
            n_pos_top = np.sum(y_true_sorted[:n_top])
            current_ef = (n_pos_top / n_top) / random_baseline
            ef_values.append(current_ef)

        plt.figure(figsize=(8, 6), dpi=600)
        plt.plot(ratios * 100, ef_values, color='#8c564b', lw=2.5, label='PhysGater Path A')
        plt.axhline(y=1.0, color='gray', linestyle=':', label='Random Screening')
        
        plt.title('Enrichment Factor (EF) Curve', fontsize=14)
        plt.xlabel('Top % of Ranked Library', fontsize=12)
        plt.ylabel('Enrichment Factor', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'ef_curve.png'), bbox_inches='tight')
        plt.close()

    def plot_tsne(self, embeddings, y_true, max_points=2000):
        """5. t-SNE 降维可视化 (采样以加快速度)"""
        if len(embeddings) == 0:
            return
            
        # English comment: see script logic.
        n_samples = len(embeddings)
        if n_samples > max_points:
            indices = np.random.choice(n_samples, max_points, replace=False)
            embeddings = embeddings[indices]
            y_true = y_true[indices]
        
        print("[Info] Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 8), dpi=600)
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, cmap='coolwarm', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Label (0=Inactive, 1=Active)')
        plt.title('t-SNE of Model Embeddings')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(os.path.join(self.output_dir, 'tsne_embedding.png'), dpi=600)
        plt.close()

def full_evaluation(model, val_loader, device, output_dir):
    print(f"[Info] Starting full evaluation and saving raw data for {output_dir}...")
    
    best_path = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(best_path):
        print("Best model weights not found.")
        return

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    y_true, y_scores, all_cls_labels = [], [], []
    all_gate_weights, all_embeddings = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Advanced Eval]"):
            ligand = batch['ligand'].to(device)
            esm = batch['esm'].to(device)
            masif = batch['masif'].to(device)
            plif = batch['plif'].to(device)
            
            # English comment: see script logic.
            logits, projection, _, _, gate_weights = model(ligand, esm, masif, plif)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            y_scores.extend(probs)
            y_true.extend(batch['label_bio'].numpy())
            all_cls_labels.extend(batch['label_cls'].numpy())
            all_gate_weights.append(gate_weights.cpu().numpy())
            all_embeddings.append(projection.cpu().numpy())
            
    # English comment: see script logic.
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    all_cls_labels = np.array(all_cls_labels)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_gate_weights = np.concatenate(all_gate_weights, axis=0)

    # English comment: see script logic.
    raw_data = {
        'y_true': y_true,
        'y_scores': y_scores,
        'all_cls_labels': all_cls_labels,
        'all_embeddings': all_embeddings,
        'all_gate_weights': all_gate_weights,
        'threshold_used': Config.eval_threshold
    }
    with open(os.path.join(output_dir, 'raw_eval_results.pkl'), 'wb') as f:
        pickle.dump(raw_data, f)

    # English comment: see script logic.
    viz = AdvancedVisualizer(output_dir)
    
    # English comment: see script logic.
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8), dpi=600)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=600)
    plt.close()

    # English comment: see script logic.
    y_pred = (y_scores > Config.eval_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inactive', 'Active'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Threshold={Config.eval_threshold})')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=600)
    plt.close()

    # English comment: see script logic.
    viz.plot_pr_curve(y_true, y_scores)
    viz.plot_score_distribution(y_true, y_scores)  # English comment removed for consistency.
    viz.plot_calibration_curve(y_true, y_scores)
    viz.plot_modality_weights([all_gate_weights])
    viz.plot_tsne(all_embeddings, y_true)
    # viz.plot_hard_negative_distribution(y_true, y_scores, all_cls_labels)
    # English comment: see script logic.
    viz.plot_enrichment_factor_curve(y_true, y_scores)
    
    print(f"[Info] All charts and raw data saved to {output_dir}")

# ==========================================
# 8. Main Execution (K-Fold Version)
# ==========================================
if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    # English comment: see script logic.
    K_FOLDS = 10
    # English comment: see script logic.
    base_output_dir = Config.output_dir 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Info] Target Output Directory: {base_output_dir}")
    print(f"[Info] Device: {device}")

    # English comment: see script logic.
    print("[Info] Loading/Processing Features...")
    plif_data = PLIFProcessor.process(Config.plif_root, Config.plif_cache)
    morgan_data = MorganProcessor.process(Config.csv_path, Config.morgan_cache, n_bits=Config.ligand_dim)
    df = pd.read_csv(Config.csv_path)

    # English comment: see script logic.
    cv_metrics = {
        'AUC': [], 'F1': [], 'Recall': [], 'Precision': [], 'MCC': [], 'FP_Rejection': []
    }
    cv_roc_data = []  # English comment removed for consistency.
    cv_pr_data = []  # English comment removed for consistency.
    
    # English comment: see script logic.
    splitter = GroupKFold(n_splits=K_FOLDS)
    target_groups = df['Target_Name']

    print(f"[Info] Starting {K_FOLDS}-Fold Cross-Validation (Threshold={Config.eval_threshold})...")

    # English comment: see script logic.
    for fold, (train_idxs, val_idxs) in enumerate(splitter.split(df, groups=target_groups)):
        print(f"\n{'='*60}")
        print(f"   FOLD {fold+1} / {K_FOLDS}")
        print(f"{'='*60}")
        
        # English comment: see script logic.
        fold_dir = os.path.join(base_output_dir, f"fold_{fold+1}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
            
        # English comment: see script logic.
        current_config = copy.deepcopy(Config)
        current_config.output_dir = fold_dir
        
        # English comment: see script logic.
        train_ds = MaSIFDataset(df.iloc[train_idxs], plif_data, morgan_data, current_config)
        val_ds = MaSIFDataset(df.iloc[val_idxs], plif_data, morgan_data, current_config)
        
        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        # English comment: see script logic.
        model = MaSIFAttentionNet(current_config).to(device)
        trainer = Trainer(model, train_loader, val_loader, current_config, device)
        
        # English comment: see script logic.
        try:
            for epoch in range(Config.epochs):
                trainer.train_epoch(epoch)
                trainer.validate(epoch) 
                
                # English comment: see script logic.
                if trainer.early_stop:
                    print(f"[Fold {fold+1}] Early stopping triggered. Moving to next fold/step.")
                    break  # English comment removed for consistency.
                    
        except KeyboardInterrupt:
            print(f"[Fold {fold+1}] Interrupted by user.")

        # English comment: see script logic.
        trainer.save_history()  # English comment removed for consistency.

        # English comment: see script logic.
        # English comment: see script logic.
        full_evaluation(model, val_loader, device, fold_dir)

        # English comment: see script logic.
        # English comment: see script logic.
        best_model_path = os.path.join(fold_dir, 'best_model.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        
        y_true_fold, y_score_fold, y_cls_fold = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, _, _, _, _ = model(batch['ligand'].to(device), batch['esm'].to(device), 
                                           batch['masif'].to(device), batch['plif'].to(device))
                y_score_fold.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                y_true_fold.extend(batch['label_bio'].numpy())
                y_cls_fold.extend(batch['label_cls'].numpy())
        
        y_true_fold = np.array(y_true_fold)
        y_score_fold = np.array(y_score_fold)
        y_cls_fold = np.array(y_cls_fold)
        y_pred_fold = (y_score_fold > Config.eval_threshold).astype(int)

        # English comment: see script logic.
        fold_auc = roc_auc_score(y_true_fold, y_score_fold)
        cv_metrics['AUC'].append(fold_auc)
        cv_metrics['F1'].append(f1_score(y_true_fold, y_pred_fold, zero_division=0))
        cv_metrics['Recall'].append(recall_score(y_true_fold, y_pred_fold, zero_division=0))
        cv_metrics['Precision'].append(precision_score(y_true_fold, y_pred_fold, zero_division=0))
        cv_metrics['MCC'].append(matthews_corrcoef(y_true_fold, y_pred_fold))
        
        fp_idx = (y_cls_fold == 2)
        cv_metrics['FP_Rejection'].append((y_score_fold[fp_idx] < Config.eval_threshold).mean() if fp_idx.sum() > 0 else 1.0)

        # English comment: see script logic.
        fpr, tpr, _ = roc_curve(y_true_fold, y_score_fold)
        prec, rec, _ = precision_recall_curve(y_true_fold, y_score_fold)
        cv_roc_data.append({'fpr': fpr, 'tpr': tpr, 'auc': fold_auc})
        cv_pr_data.append({'precision': prec, 'recall': rec})

        # English comment: see script logic.
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # English comment: see script logic.
    # ==========================================
    print(f"\n{'-'*60}\nCROSS-VALIDATION COMPLETE\n{'-'*60}")
    summary_dir = os.path.join(base_output_dir, "cv_summary")
    if not os.path.exists(summary_dir): os.makedirs(summary_dir)

    # English comment: see script logic.
    with open(os.path.join(summary_dir, 'cv_summary_raw_data.pkl'), 'wb') as f:
        pickle.dump({'metrics': cv_metrics, 'roc': cv_roc_data, 'pr': cv_pr_data}, f)

    # English comment: see script logic.
    with open(os.path.join(summary_dir, "metrics_summary.txt"), "w") as f:
        for m, values in cv_metrics.items():
            line = f"{m}: Mean={np.mean(values):.4f} ± {np.std(values):.4f}"
            print(line)
            f.write(line + "\n")

    # English comment: see script logic.
    plot_cross_validation_roc(cv_roc_data, summary_dir)  # English comment removed for consistency.
    plot_cross_validation_pr(cv_pr_data, summary_dir)  # English comment removed for consistency.
    plot_cv_metrics_summary(cv_metrics, summary_dir)  # English comment removed for consistency.
    plot_cv_radar_chart(cv_metrics, summary_dir)  # English comment removed for consistency.
    
    print(f"\n[Done] All cross-validation results are saved in: {base_output_dir}")