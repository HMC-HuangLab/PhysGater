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
import pandas as pd
import numpy as np
# English comment: see script logic.
import prolif as plf
from rdkit import Chem
from tqdm import tqdm
import warnings

# English comment: see script logic.
warnings.filterwarnings('ignore')

def read_pdbqt_as_rdkit(pdbqt_path):
    """
    读取 PDBQT 文件，修复 Vina 原子类型 (A->C)，并以 sanitize=False 模式加载。
    """
    if not os.path.exists(pdbqt_path):
        return None
    
    try:
        with open(pdbqt_path, 'r') as f:
            lines = f.readlines()
        
        pdb_block = ""
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                if len(line) > 77:
                    atom_type = line[76:78].strip()
                    # English comment: see script logic.
                    if atom_type == 'A': 
                        line = line[:76] + ' C' + line[78:]
                    elif atom_type == 'NA':
                        line = line[:76] + ' N' + line[78:]
                    elif atom_type == 'OA':
                        line = line[:76] + ' O' + line[78:]
                    elif atom_type == 'SA':
                        line = line[:76] + ' S' + line[78:]
                    elif atom_type == 'HD':
                        line = line[:76] + ' H' + line[78:]
                pdb_block += line
            elif line.startswith('ENDMDL'):
                break
            elif line.startswith('CONECT'):
                pdb_block += line
        
        # English comment: see script logic.
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
        if mol:
            try:
                mol.UpdatePropertyCache(strict=False)
            except:
                pass
        return mol
    except Exception:
        return None

def load_protein_safe(prot_path):
    """
    [核心修复] 使用 RDKit 直接加载蛋白 PDB，避开 MDAnalysis 的键推断错误。
    """
    if not os.path.exists(prot_path):
        return None
        
    try:
        # English comment: see script logic.
        # English comment: see script logic.
        mol = Chem.MolFromPDBFile(prot_path, removeHs=False, sanitize=False)
        
        if mol is None:
            # English comment: see script logic.
            return None

        # English comment: see script logic.
        try:
            mol.UpdatePropertyCache(strict=False)
            # English comment: see script logic.
            # English comment: see script logic.
            Chem.GetSymmSSSR(mol) 
        except:
            pass
            
        # English comment: see script logic.
        return plf.Molecule.from_rdkit(mol)
        
    except Exception as e:
        print(f"  [Protein Load Error] {os.path.basename(prot_path)}: {e}")
        return None

def find_protein_file(protein_root, target_name):
    """
    模糊查找蛋白文件: {target_name}_*.pdb
    """
    pattern = os.path.join(protein_root, f"{target_name}_*.pdb")
    candidates = glob.glob(pattern)
    if candidates:
        return candidates[0]
    # English comment: see script logic.
    exact = os.path.join(protein_root, f"{target_name}.pdb")
    if os.path.exists(exact):
        return exact
    return None

def generate_plif_robust(csv_file, ligands_root, protein_root, output_root):
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    print(f"正在读取 CSV: {csv_file} ...")
    df = pd.read_csv(csv_file)
    grouped = df.groupby('Target_Name')
    
    # English comment: see script logic.
    interactions = [
        "Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", 
        "Anionic", "Cationic", "CationPi", "PiCation"
    ]
    fp = plf.Fingerprint(interactions)
    
    print(f"开始计算 PLIF (RDKit Loading Mode)...")
    
    for target_name, group_df in tqdm(grouped, desc="Processing Targets"):
        
        # English comment: see script logic.
        prot_path = find_protein_file(protein_root, target_name)
        if not prot_path:
            print(f"\n[跳过] 找不到蛋白文件: {target_name}")
            continue
            
        # English comment: see script logic.
        prot_mol = load_protein_safe(prot_path)
        if not prot_mol:
            print(f"\n[跳过] 蛋白加载失败 (RDKit Parse Error): {target_name}")
            continue
            
        # English comment: see script logic.
        lig_dir = os.path.join(ligands_root, target_name)
        if not os.path.exists(lig_dir):
            continue  # English comment removed for consistency.
            
        # English comment: see script logic.
        valid_mols = []
        valid_names = []
        
        for idx, row in group_df.iterrows():
            mol_name = row['Name']
            mol_path = os.path.join(lig_dir, f"{mol_name}.pdbqt")
            rdkit_mol = read_pdbqt_as_rdkit(mol_path)
            if rdkit_mol:
                valid_mols.append(plf.Molecule.from_rdkit(rdkit_mol))
                valid_names.append(mol_name)
        
        if not valid_mols:
            print(f"\n[警告] 靶点 {target_name} 无有效配体。")
            continue

        # English comment: see script logic.
        try:
            ifp = fp.run_from_iterable(valid_mols, prot_mol)
            df_ifp = fp.to_dataframe()
            
            # English comment: see script logic.
            df_ifp.index = valid_names
            new_columns = [f"{col[0]}_{col[1]}" for col in df_ifp.columns]
            df_ifp.columns = new_columns
            df_ifp = df_ifp.astype(int)
            
            df_ifp.reset_index(inplace=True)
            df_ifp.rename(columns={'index': 'Name'}, inplace=True)
            
            save_path = os.path.join(output_root, f"{target_name}_plif_features.csv")
            df_ifp.to_csv(save_path, index=False)
            
        except Exception as e:
            # English comment: see script logic.
            # English comment: see script logic.
            print(f"\n[错误] 计算 {target_name} 时崩溃: {e}")

    print("所有任务完成。")

# English comment: see script logic.
if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    # English comment: see script logic.
    CSV_FILE = "/mnt/data/fpdetec_V2/final_with_thresholds/new_train_data_relabelled.csv"
    
    # English comment: see script logic.
    LIGAND_ROOT = "./Dataset_Structured_PDBQT"
    
    # English comment: see script logic.
    PROTEIN_ROOT = "/mnt/data/Kinase/table/kinase_protein//" 
    
    # English comment: see script logic.
    OUTPUT_ROOT = "./Dataset_PLIF_Flat"
    
    generate_plif_robust(CSV_FILE, LIGAND_ROOT, PROTEIN_ROOT, OUTPUT_ROOT)


