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
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle

def generate_morgan_fingerprints(csv_path, save_path, radius=2, n_bits=2048):
    df = pd.read_csv(csv_path)
    morgan_dict = {}
    
    print("Generating Morgan Fingerprints...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = row['Name']
        smiles = row['smiles']  # English comment removed for consistency.
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # English comment: see script logic.
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                # English comment: see script logic.
                arr = np.zeros((1,), dtype=np.float32)
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                morgan_dict[name] = arr
            else:
                # English comment: see script logic.
                morgan_dict[name] = np.zeros((n_bits,), dtype=np.float32)
        except:
            morgan_dict[name] = np.zeros((n_bits,), dtype=np.float32)
            
    print(f"Saving to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(morgan_dict, f)

# English comment: see script logic.
generate_morgan_fingerprints("/mnt/data/fpdetec_V2/final_with_thresholds/new_train_data_relabelled.csv", "morgan_2048_cache.pkl")