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
import torch
import copy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
# English comment: see script logic.
from pathA_final_masif_model import MaSIFAttentionNet, Trainer, MaSIFDataset, Config
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from scipy.stats import rankdata

def calculate_ef(y_true, y_scores, top_ratio=0.01):
    """计算富集因子 (Enrichment Factor)"""
    n_total = len(y_true)
    n_pos_total = np.sum(y_true)
    if n_pos_total == 0: return 0.0
    n_top = max(1, int(n_total * top_ratio))
    
    indices = np.argsort(y_scores)[::-1]
    top_indices = indices[:n_top]
    n_pos_top = np.sum(y_true[top_indices])
    return (n_pos_top / n_top) / (n_pos_total / n_total)

def calculate_bedroc(y_true, y_scores, alpha=20.0):
    """
    计算 BEDROC (Boltzmann-Enhanced Discrimination of ROC)
    alpha=20.0 是工业界标准，衡量前 8% 的富集能力
    """
    n = len(y_true)
    n_pos = np.sum(y_true)
    if n_pos == 0 or n_pos == n: return 0.0
    
    # English comment: see script logic.
    ranks = n - rankdata(y_scores, method='average') + 1
    pos_ranks = ranks[y_true == 1]
    
    ins_sum = np.sum(np.exp(-alpha * pos_ranks / n))
    r_a = n_pos / n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    max_sum = (1 - np.exp(-alpha * r_a)) / (1 - np.exp(-alpha / n))
    
    return ins_sum / max_sum  # English comment removed for consistency.

def collect_ablation_raw_data(model, loader, device):
    """在验证集上跑一遍，收集所有原始信息"""
    model.eval()
    all_scores, all_labels = [], []
    all_weights = []
    
    with torch.no_grad():
        for batch in loader:
            l, e, m, pl = batch['ligand'].to(device), batch['esm'].to(device), \
                         batch['masif'].to(device), batch['plif'].to(device)
            # English comment: see script logic.
            logits, _, _, _, weights = model(l, e, m, pl)
            all_scores.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(batch['label_bio'].numpy())
            all_weights.append(weights.cpu().numpy()) # [Batch, 5]

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_weights = np.concatenate(all_weights, axis=0).mean(axis=0)  # English comment removed for consistency.
    
    # English comment: see script logic.
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    prec, rec, _ = precision_recall_curve(all_labels, all_scores)
    
    # English comment: see script logic.
    interp_tpr = np.interp(np.linspace(0, 1, 100), fpr, tpr)
    interp_prec = np.interp(np.linspace(0, 1, 100), rec[::-1], prec[::-1])
    
    return {
        'scores': all_scores,
        'labels': all_labels,
        'curve_tpr': interp_tpr,
        'curve_prec': interp_prec,
        'avg_weights': all_weights,
        'ef1': calculate_ef(all_labels, all_scores, 0.01),
        'bedroc': calculate_bedroc(all_labels, all_scores)
    }
# ==========================================
# English comment: see script logic.
# ==========================================
class AblationConfig(Config):
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    RUN_LIST = ['simple_concat', 'no_contrastive', 'no_ranking']  # English comment removed for consistency.
    
    base_output_root = "/mnt/data/fpdetec_V2/Ablation_Study_Results2"
    epochs = 50
    batch_size = 64

# ==========================================
# English comment: see script logic.
# ==========================================
class AblationMaSIFAttentionNet(MaSIFAttentionNet):
    def __init__(self, cfg, mode='full'):
        super().__init__(cfg)
        self.mode = mode
        # English comment: see script logic.
        if mode == 'simple_concat':
            # English comment: see script logic.
            # English comment: see script logic.
            self.fusion_dim = cfg.hidden_dim * 6  # English comment removed for consistency.
            
    def forward(self, ligand, esm, masif, plif):
        # English comment: see script logic.
        if self.mode == 'no_plif':
            plif = torch.zeros_like(plif)
        elif self.mode == 'no_masif':
            masif = torch.zeros_like(masif)
        elif self.mode == 'no_esm2':
            esm = torch.zeros_like(esm)
        elif self.mode == 'fingerprint_only':
            esm = torch.zeros_like(esm)
            masif = torch.zeros_like(masif)
            plif = torch.zeros_like(plif)
            # English comment: see script logic.

        # English comment: see script logic.
        l_v = self.ligand_fc(ligand)
        m_p, m_g = self.masif_encoder(masif)
        i_v = self.masif_attn(l_v, m_p)
        e_v = self.esm_fc(esm)
        p_v = self.plif_fc(plif)

        # English comment: see script logic.
        if self.mode == 'simple_concat':
            # English comment: see script logic.
            f_v = (l_v + i_v + e_v + p_v + m_g) / 5.0
            gate_weights = torch.ones((ligand.size(0), 5), device=ligand.device) * 0.2
        else:
            f_v, gate_weights = self.gated_fusion([l_v, i_v, e_v, p_v, m_g])

        combined = torch.cat([f_v, l_v, i_v, e_v, p_v, m_g], dim=1)
        logits = self.classifier(combined)
        
        # English comment: see script logic.
        projection = self.projector(combined)
        lig_ctx = self.ligand_ctx(l_v)
        prot_ctx = self.protein_ctx(torch.cat([i_v, e_v, m_g], dim=1))
        
        return logits, projection, lig_ctx, prot_ctx, gate_weights

# ==========================================
# English comment: see script logic.
# ==========================================
def run_ablation_study(mode, plif_data, morgan_data):
    print(f"\n{'#'*40}")
    print(f"🚀 启动消融实验项目: {mode}")
    print(f"{'#'*40}")
    
    # English comment: see script logic.
    mode_dir = os.path.join(AblationConfig.base_output_root, mode)
    if not os.path.exists(mode_dir): os.makedirs(mode_dir)

    # English comment: see script logic.
    df = pd.read_csv(AblationConfig.csv_path)
    splitter = GroupKFold(n_splits=10)
    
    # English comment: see script logic.
    fold_metrics = []
    fold_raw_data = []

    # English comment: see script logic.
    # English comment: see script logic.
    for fold, (t_idx, v_idx) in enumerate(splitter.split(df, groups=df['Target_Name']), 1):
        print(f"\n>>> [实验: {mode}] 正在执行第 {fold} / 10 折...")
        
        fold_dir = os.path.join(mode_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir): os.makedirs(fold_dir)
        
        # English comment: see script logic.
        cfg = copy.deepcopy(AblationConfig)
        cfg.output_dir = fold_dir
        if mode == 'no_contrastive': cfg.contrastive_weight = 0.0
        if mode == 'no_ranking': cfg.ranking_weight = 0.0

        # English comment: see script logic.
        train_ds = MaSIFDataset(df.iloc[t_idx], plif_data, morgan_data, cfg)
        val_ds = MaSIFDataset(df.iloc[v_idx], plif_data, morgan_data, cfg)
        
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        # English comment: see script logic.
        model = AblationMaSIFAttentionNet(cfg, mode=mode).to(device)
        trainer = Trainer(model, train_loader, val_loader, cfg, device)
        
        # English comment: see script logic.
        for epoch in range(cfg.epochs):
            trainer.train_epoch(epoch)
            # English comment: see script logic.
            f1, auc_score, fp_rej = trainer.validate(epoch)
        
        # English comment: see script logic.
        print(f"--- 第 {fold} 折训练完成，正在提取原始性能数据... ---")
        best_model_path = os.path.join(fold_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # English comment: see script logic.
        raw_info = collect_ablation_raw_data(model, val_loader, device)
        fold_raw_data.append(raw_info)

        fold_metrics.append({
            'fold': fold, 
            'AUC': roc_auc_score(raw_info['labels'], raw_info['scores']), 
            'EF1%': raw_info['ef1'],
            'BEDROC': raw_info['bedroc'],
            'F1': f1, 
            'FP_Rej': fp_rej
        })

        # English comment: see script logic.
        del model, trainer, train_ds, val_ds
        torch.cuda.empty_cache()

    # ================================================================
    # English comment: see script logic.
    # ================================================================
    print(f"\n📉 [汇总] 正在计算 {mode} 的十折平均表现并持久化...")
    
    import pickle
    summary_pack = {
        'mode': mode,
        'fold_metrics': fold_metrics,
        # English comment: see script logic.
        'mean_tpr': np.mean([x['curve_tpr'] for x in fold_raw_data], axis=0),
        'mean_prec': np.mean([x['curve_prec'] for x in fold_raw_data], axis=0),
        # English comment: see script logic.
        'modality_importance': np.mean([x['avg_weights'] for x in fold_raw_data], axis=0)
    }
    
    # English comment: see script logic.
    with open(os.path.join(mode_dir, "ablation_full_data.pkl"), "wb") as f:
        pickle.dump(summary_pack, f)

    # English comment: see script logic.
    pd.DataFrame(fold_metrics).to_csv(os.path.join(mode_dir, "cv_metrics.csv"), index=False)
    print(f"✅ 模式 {mode} 全部十折任务已完成！结果存至: {mode_dir}")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    _apply_overrides_to_class(AblationConfig, _ovr)
    # English comment: see script logic.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # English comment: see script logic.
    print("[Info] Pre-loading Features...")
    from pathA_final_masif_model import PLIFProcessor, MorganProcessor
    
    plif_data = PLIFProcessor.process(AblationConfig.plif_root, AblationConfig.plif_cache)
    morgan_data = MorganProcessor.process(AblationConfig.csv_path, AblationConfig.morgan_cache, 
                                          n_bits=AblationConfig.ligand_dim)

    # English comment: see script logic.
    for mode in AblationConfig.RUN_LIST:
        run_ablation_study(mode, plif_data, morgan_data)  # English comment removed for consistency.