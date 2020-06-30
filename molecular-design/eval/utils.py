import os
from typing import List, Optional, Tuple, Union

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from chemprop.data.utils import get_smiles
from chemprop.features import get_features_generator

UNCONDITIONAL_EVAL_DUMMY = '<DUMMY>'


SMILES_TO_MORGAN = {}


morgan_fingerprint = get_features_generator('morgan')


def get_pred_smiles(pred_smiles_dir: str, return_num_decode: bool = False) -> Union[List[str], Tuple[List[str], int]]:
    # Collect all paths to .txt files
    pred_paths = []
    for root, _, files in os.walk(pred_smiles_dir):
        pred_paths += [os.path.join(root, fname) for fname in files if fname.endswith('.txt')]

    # Get SMILES
    all_smiles = []
    for path in pred_paths:
        all_smiles.append(get_smiles(path, header=False))

    # Check that each list of smiles loaded from a file has the same number of smiles
    sizes = {len(smiles) for smiles in all_smiles}
    assert len(sizes) == 1
    num_examples = sizes.pop()
    num_decode = len(all_smiles)

    # Reorder smiles so that all translations of a source molecule are contiguous
    smiles = [all_smiles[j][i] for i in range(num_examples) for j in range(num_decode)]

    # Filter out invalid smiles
    num_tot_smiles = sum(1 for smile in smiles if smile is not None)  # Shouldn't be any Nones but just to be safe
    smiles = replace_invalid_smiles(smiles)
    num_valid_smiles = sum(1 for smile in smiles if smile is not None)
    print(f'Valid smiles = {100 * num_valid_smiles / num_tot_smiles}% ({num_valid_smiles}/{num_tot_smiles})')

    # Optionally return the number of decodings for each source molecule
    if return_num_decode:
        return smiles, num_decode

    return smiles


def get_train_smiles(train_path: str) -> List[str]:
    smiles = []
    with open(train_path) as f:
        for line in f:
            smiles += line.strip().split()

    return smiles


def get_morgan_with_cache(smiles):
    if smiles in SMILES_TO_MORGAN:
        fp = SMILES_TO_MORGAN[smiles]
    else:
        amol = Chem.MolFromSmiles(smiles)
        if amol is None:
            fp = None
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        SMILES_TO_MORGAN[smiles] = fp
    return fp


def tanimoto_similarity(smiles_1: str, smiles_2: str) -> float:
    if UNCONDITIONAL_EVAL_DUMMY in smiles_1 or UNCONDITIONAL_EVAL_DUMMY in smiles_2:
        return 1.0

    if smiles_1 is None or smiles_2 is None:
        return 0.0
    
    fp1, fp2 = get_morgan_with_cache(smiles_1), get_morgan_with_cache(smiles_2)

    if fp1 is None or fp2 is None:
        return 0.0

    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def clear_fp_cache():
    global SMILES_TO_MORGAN
    SMILES_TO_MORGAN = {}


def replace_invalid_smiles(smiles: List[str]) -> List[Optional[str]]:
    """Replaces invalid smiles with None."""
    return [smile if Chem.MolFromSmiles(smile) is not None else None for smile in tqdm(smiles, total=len(smiles))]
