import torch


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
import esm
import numpy as np
from Bio.PDB import PDBParser
import warnings
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

# --- Global Settings & Warning Suppression ---
warnings.filterwarnings("ignore", ".*A segmentation fault may occur.*")
warnings.filterwarnings("ignore", ".*Regression weights not found.*")
warnings.filterwarnings("ignore", ".*PDBConstructionWarning.*")
# Global device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Module 1: Identify Residues from Pocket PDB
# ==============================================================================
def get_pocket_residues_from_pdb(pocket_pdb_path: str):
    """Robustly identifies all residues and their chain ID from a pocket PDB file."""
    if not os.path.exists(pocket_pdb_path): return None, None
    unique_residues = set()
    try:
        with open(pocket_pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        chain_id, residue_seq_num = line[21], int(line[22:26].strip())
                        unique_residues.add((chain_id, residue_seq_num))
                    except (IndexError, ValueError): continue
    except Exception: return None, None
    
    if not unique_residues: return None, None
    
    pocket_info = defaultdict(list)
    for chain, res_id in unique_residues: pocket_info[chain].append(res_id)
    
    # Assume the pocket is on a single chain, use the first one found.
    chain_id_to_use = list(pocket_info.keys())[0]
    residue_ids = sorted(pocket_info[chain_id_to_use])
    return chain_id_to_use, residue_ids

# ==============================================================================
# Module 2: ESM-2 Feature Extraction
# ==============================================================================
def extract_full_chain_esm_features(pdb_path: str, chain_id: str, esm_model, esm_alphabet):
    """Extracts sequence features for an entire chain using a pre-loaded ESM-2 model."""
    THREE_TO_ONE = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
                    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
                    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        if chain_id not in structure[0]: raise ValueError(f"Chain '{chain_id}' not in PDB")
        
        chain = structure[0][chain_id]
        # Filter for standard residues that have a C-alpha atom
        valid_residues = [res for res in chain if res.id[0] == ' ' and 'CA' in res]
        
        seq, residue_ids_map = [], {}
        for i, residue in enumerate(valid_residues):
            res_name = residue.get_resname()
            if res_name in THREE_TO_ONE:
                seq.append(THREE_TO_ONE[res_name])
                # Map the residue sequence number (e.g., 155) to its index in the sequence (e.g., 0, 1, 2...)
                residue_ids_map[residue.id[1]] = i
            
        native_seq = ''.join(seq)
        if not native_seq: return None, None
        
        batch_converter = esm_alphabet.get_batch_converter()
        with torch.no_grad():
            _, _, batch_tokens = batch_converter([(chain_id, native_seq)])
            # Use layer 33 for esm2_t33_650M_UR50D
            results = esm_model(batch_tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0]
            # Remove start and end tokens [CLS] and [SEP]
            sequence_representations = token_representations[1:len(native_seq)+1].cpu()
        return sequence_representations, residue_ids_map
    except Exception as e:
        print(f"❌ Error during ESM-2 feature extraction for {pdb_path}: {e}")
        return None, None

# ==============================================================================
# Module 3: Main Processing Logic
# ==============================================================================
def find_protein_file_pairs(input_dir: str):
    """Finds matching pairs of {base}.pdb (full) and {base}_15A.pdb (pocket) files."""
    print(f"🔍 Scanning input directory: {input_dir}")
    file_groups = defaultdict(dict)
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_15A.pdb"):
            base_name = filename.replace("_15A.pdb", "")
            file_groups[base_name]['pocket_pdb'] = os.path.join(input_dir, filename)
        elif filename.endswith(".pdb"):
            base_name = os.path.splitext(filename)[0]
            file_groups[base_name]['full_pdb'] = os.path.join(input_dir, filename)

    valid_pairs = []
    for name, files in file_groups.items():
        if 'full_pdb' in files and 'pocket_pdb' in files:
            files['name'] = name
            valid_pairs.append(files)
            
    print(f"✅ Found {len(valid_pairs)} valid pairs of full and pocket PDB files.")
    return valid_pairs

def process_protein(protein_pair, output_dir, esm_model, esm_alphabet):
    """Processes a single protein file pair to generate pooled pocket ESM features."""
    name = protein_pair['name']
    output_file = os.path.join(output_dir, f"{name}.pt")
    
    if os.path.exists(output_file):
        return ("Skipped", f"{name} already exists")

    try:
        # Step 1: Identify pocket residues from the pocket PDB
        chain_id, pocket_residue_ids = get_pocket_residues_from_pdb(protein_pair['pocket_pdb'])
        if not chain_id or not pocket_residue_ids:
            return ("Failed", f"{name}: Could not identify any pocket residues")

        # Step 2: Extract ESM-2 features for the entire protein chain
        full_chain_esm_features, residue_map = extract_full_chain_esm_features(
            protein_pair['full_pdb'], chain_id, esm_model, esm_alphabet
        )
        if full_chain_esm_features is None:
            return ("Failed", f"{name}: ESM-2 feature extraction failed")

        # Step 3: Select only the features for the pocket residues
        pocket_esm_indices = []
        for res_id in pocket_residue_ids:
            if res_id in residue_map:
                esm_idx = residue_map[res_id]
                pocket_esm_indices.append(esm_idx)
        
        if not pocket_esm_indices:
            return ("Failed", f"{name}: Could not map any pocket residues to the ESM sequence")
            
        pocket_esm_features = full_chain_esm_features[pocket_esm_indices]

        # Step 4: Pool the pocket residue features into a single fixed-size vector
        if pocket_esm_features.shape[0] == 0:
            return ("Failed", f"{name}: Pocket feature tensor is empty after selection")

        # Perform the pooling (e.g., mean + max)
        mean_pooled = torch.mean(pocket_esm_features, dim=0)
        max_pooled = torch.max(pocket_esm_features, dim=0).values
        final_pocket_vector = torch.cat((mean_pooled, max_pooled), dim=0) # Shape should be [2560]

        # --- ENSURE THIS IS THE LINE THAT SAVES ---
        # Step 5: Save ONLY the final pooled feature vector (a 1D tensor)
        torch.save(final_pocket_vector, output_file)
        # --- DO NOT SAVE pocket_esm_features ---

        return ("Success", f"{name} processed successfully")
        

    except Exception as e:
        return ("Failed", f"An unexpected error occurred while processing {name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Protein Pocket ESM-2 Feature Extraction Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing all .pdb and _15A.pdb files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the final feature .pt files.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("⏳ Loading ESM-2 model, this may take a moment...")
    try:
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_model.eval().to(DEVICE)
        print(f"✅ ESM-2 model loaded successfully onto {DEVICE}")
    except Exception as e:
        print(f"❌ Could not load ESM-2 model. Check your 'esm' installation and internet connection: {e}")
        return

    protein_pairs = find_protein_file_pairs(args.input_dir)
    if not protein_pairs: return
    
    results = {"Success": [], "Skipped": [], "Failed": []}
    
    with tqdm(total=len(protein_pairs), desc="Overall Progress") as pbar:
        for protein_pair in protein_pairs:
            status, message = process_protein(protein_pair, args.output_dir, esm_model, esm_alphabet)
            results[status].append(message)
            pbar.update(1)
            pbar.set_postfix_str(f"Latest: {message}")

    print("\n" + "="*30 + " Processing Summary " + "="*30)
    print(f"✅ Successful: {len(results['Success'])} proteins")
    print(f"⏩ Skipped (already exist): {len(results['Skipped'])} proteins")
    print(f"❌ Failed: {len(results['Failed'])} proteins")
    if results['Failed']:
        print("\n--- Failure Details ---")
        for i, reason in enumerate(results['Failed']):
            print(f"{i+1}. {reason}")
    print("="*82)

if __name__ == '__main__':
    main()