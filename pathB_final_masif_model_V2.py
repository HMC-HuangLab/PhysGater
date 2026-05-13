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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, matthews_corrcoef, 
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import shutil
import warnings
import gc
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit is not installed.")

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # --- Paths ---
    csv_path = "/mnt/data/fpdetec_V2/final_with_thresholds/new_train_data_relabelled.csv"
    esm2_root = "/mnt/data/fpdetec_V2/new_esm2_feat"
    morgan_cache = "/mnt/data/fpdetec_V2/morgan_2048_cache.pkl"

    # --- Output ---
    output_dir = "/mnt/data/fpdetec_V2/PathB_Hunter_Corrected_v3" 

    # --- Dimensions ---
    ligand_dim = 2048
    esm2_dim = 2560
    hidden_dim = 512 
    
    # --- Training ---
    k_folds = 10
    batch_size = 128
    
    # English comment: see script logic.
    lr = 2e-5        
    epochs = 50
    
    # English comment: see script logic.
    pos_weight = 1.0  
    
    clip_grad_norm = 1.0
    eval_threshold = 0.5 
    
    ema_decay = 0.999
    use_amp = True
    dropout_prob = 0.4  # English comment removed for consistency.
    seed = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(Config.seed)

# ==========================================
# 2. Processors
# ==========================================
class MorganProcessor:
    @staticmethod
    def process(csv_path, cache_path, n_bits=2048):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f: return pickle.load(f)
        
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
                
        with open(cache_path, 'wb') as f: pickle.dump(morgan_dict, f)
        return morgan_dict

# ==========================================
# 3. Dataset
# ==========================================
class PathBDataset(Dataset):
    def __init__(self, df, morgan_dict, config):
        self.df = df.reset_index(drop=True)
        self.morgan_dict = morgan_dict
        self.cfg = config
        self.pid_col = 'protien_id' if 'protien_id' in df.columns else 'protein_id'
        
        print("[Info] Preloading ESM Features...")
        self.esm_cache = {}
        unique_pids = self.df[self.pid_col].unique()
        
        for pid in tqdm(unique_pids, desc="Loading ESM", leave=False):
            pid_str = str(pid).strip()
            esm_path = os.path.join(config.esm2_root, f"{pid_str}.pt")
            esm_val = torch.zeros(config.esm2_dim)
            if os.path.exists(esm_path):
                try:
                    t = torch.load(esm_path, map_location='cpu')
                    if t.dim() == 2: t = t.mean(dim=0)
                    esm_val = t
                except: pass
            self.esm_cache[pid_str] = esm_val

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row['Name'])
        pid_str = str(row[self.pid_col]).strip()
        
        ligand = torch.tensor(self.morgan_dict.get(name, np.zeros(self.cfg.ligand_dim)), dtype=torch.float32)
        esm = self.esm_cache.get(pid_str, torch.zeros(self.cfg.esm2_dim))
        
        return {
            'ligand': ligand,
            'esm': esm,
            'label': torch.tensor(float(row['Ground_Truth']), dtype=torch.float32)
        }

# ==========================================
# 4. Model
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.net(x))

class PathB_Hunter_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.ligand_encoder = nn.Sequential(
            nn.Linear(cfg.ligand_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_prob),
            ResidualBlock(cfg.hidden_dim, cfg.dropout_prob)
        )
        
        self.esm_encoder = nn.Sequential(
            nn.Linear(cfg.esm2_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_prob),
            ResidualBlock(cfg.hidden_dim, cfg.dropout_prob)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
        )
        
    def forward(self, ligand, esm):
        l_emb = self.ligand_encoder(ligand)
        p_emb = self.esm_encoder(esm)
        combined_embedding = torch.cat([l_emb, p_emb], dim=1)
        logits = self.classifier(combined_embedding)
        return logits, combined_embedding

# ==========================================
# 5. Visualization
# ==========================================
class AdvancedVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)

    def plot_pr_curve(self, y_true, y_scores):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#2ca02c', lw=2, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'))
        plt.close()

    def plot_score_distribution(self, y_true, y_scores):
        df = pd.DataFrame({'Score': y_scores, 'Label': y_true})
        df['Label'] = df['Label'].map({0: 'Inactive', 1: 'Active'})
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='Score', hue='Label', bins=40, kde=True, element="step", stat="density", common_norm=False, palette=['#1f77b4', '#d62728'])
        plt.axvline(x=Config.eval_threshold, color='black', linestyle='--', label=f'Thr={Config.eval_threshold}')
        plt.title('Prediction Score Distribution')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'))
        plt.close()

    def plot_tsne(self, embeddings, y_true, max_points=3000):
        if len(embeddings) == 0: return
        n_samples = len(embeddings)
        if n_samples > max_points:
            indices = np.random.choice(n_samples, max_points, replace=False)
            embeddings = embeddings[indices]
            y_true = y_true[indices]
        
        print("  Running t-SNE on Embeddings...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, cmap='coolwarm', alpha=0.6, s=15)
        plt.colorbar(scatter, label='Label (0=Inactive, 1=Active)')
        plt.title('t-SNE of PathB Embeddings (Ligand+Seq)')
        plt.savefig(os.path.join(self.output_dir, 'tsne_embedding.png'))
        plt.close()

def plot_fold_charts(trainer, y_true, y_score, output_dir):
    epochs = range(1, len(trainer.history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, trainer.history['train_loss'], 'k-', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, trainer.history['val_recall'], 'b-o', label='Recall')
    ax2.plot(epochs, trainer.history['val_f1'], 'g--', label='F1')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    y_pred = (y_score > Config.eval_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=['Inactive', 'Active']).plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (Thr={Config.eval_threshold})')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def full_evaluation(model, val_loader, device, output_dir):
    best_path = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(best_path): return

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    y_true, y_scores = [], []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Full Eval"):
            l = batch['ligand'].to(device)
            e = batch['esm'].to(device)
            logits, feats = model(l, e)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_scores.extend(probs)
            y_true.extend(batch['label'].numpy())
            all_embeddings.append(feats.cpu().numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    viz = AdvancedVisualizer(output_dir)
    viz.plot_pr_curve(y_true, y_scores)
    viz.plot_score_distribution(y_true, y_scores)
    viz.plot_tsne(all_embeddings, y_true)

# ==========================================
# 6. Trainer (Standard BCE + Weighted Sampling)
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        
        # English comment: see script logic.
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.scaler = GradScaler(enabled=config.use_amp)
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters(): param.requires_grad = False
        
        if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)
        self.best_recall = 0.0
        self.history = {'train_loss': [], 'val_recall': [], 'val_f1': []}

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.cfg.ema_decay).add_(param.data, alpha=1 - self.cfg.ema_decay)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}")
        
        for batch in loop:
            ligand = batch['ligand'].to(self.device)
            esm = batch['esm'].to(self.device)
            label = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            with autocast(enabled=self.scaler.is_enabled()):
                logits, _ = self.model(ligand, esm)
                loss = self.criterion(logits.view(-1), label)
            
            if torch.isnan(loss):
                self.scaler.update()
                continue

            self.scaler.scale(loss).backward()
            
            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_ema()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        self.scheduler.step()
        return avg_loss

    def validate(self, epoch):
        model_to_eval = self.ema_model
        model_to_eval.eval()
        y_true, y_probs = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                ligand = batch['ligand'].to(self.device)
                esm = batch['esm'].to(self.device)
                label = batch['label'].numpy()
                
                logits, _ = model_to_eval(ligand, esm)
                probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
                
                y_true.extend(label)
                y_probs.extend(probs)
                
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        
        y_pred = (y_probs > self.cfg.eval_threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_val = roc_auc_score(y_true, y_probs)
        
        print(f"  [Val] Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f}")
        
        if rec > self.best_recall:
            self.best_recall = rec
            torch.save(model_to_eval.state_dict(), os.path.join(self.cfg.output_dir, "best_model.pth"))
            print(f"  *** New Best Recall Model Saved ***")
            
        self.history['val_recall'].append(rec)
        self.history['val_f1'].append(f1)
        
        return rec, f1, auc_val, y_true, y_probs

# ==========================================
# 7. Main Execution (K-Fold)
# ==========================================
if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(Config, _ovr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    morgan_data = MorganProcessor.process(Config.csv_path, Config.morgan_cache, Config.ligand_dim)
    df = pd.read_csv(Config.csv_path)
    
    if not os.path.exists(Config.output_dir): os.makedirs(Config.output_dir)
    
    splitter = GroupKFold(n_splits=10)
    print(f"Starting {Config.k_folds}-Fold CV for Path B Hunter Model (Stabilized)...")
    
    # K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, groups=df['Target_Name'])):
        print(f"\n{'='*40}")
        print(f" Fold {fold+1} / {Config.k_folds}")
        print(f"{'='*40}")
        
        fold_dir = os.path.join(Config.output_dir, f"fold_{fold+1}")
        current_config = copy.deepcopy(Config)
        current_config.output_dir = fold_dir
        
        train_ds = PathBDataset(df.iloc[train_idx], morgan_data, Config)
        val_ds = PathBDataset(df.iloc[val_idx], morgan_data, Config)
        
        # Weighted Sampler (Key for Imbalance)
        train_labels = df.iloc[train_idx]['Ground_Truth'].values.astype(int)
        class_counts = np.bincount(train_labels)
        class_counts = np.where(class_counts == 0, 1, class_counts) 
        weights = 1. / class_counts
        samples_weights = weights[train_labels]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
        
        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4)
        
        model = PathB_Hunter_Net(current_config).to(device)
        trainer = Trainer(model, train_loader, val_loader, current_config, device)
        
        y_true_final, y_score_final = [], []
        try:
            for epoch in range(Config.epochs):
                trainer.train_epoch(epoch)
                rec, f1, auc_val, yt, ys = trainer.validate(epoch)
                y_true_final, y_score_final = yt, ys
        except KeyboardInterrupt:
            print("Stopped.")
            
        print(f"  [Fold {fold+1}] Generating charts...")
        plot_fold_charts(trainer, y_true_final, y_score_final, fold_dir)
        full_evaluation(model, val_loader, device, fold_dir)
        
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    print(f"Done. Check {Config.output_dir}")