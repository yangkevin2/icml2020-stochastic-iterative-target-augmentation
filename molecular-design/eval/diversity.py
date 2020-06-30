from argparse import ArgumentParser
from typing import List, Optional, Tuple

import numpy as np

from utils import get_pred_smiles, tanimoto_similarity, clear_fp_cache, UNCONDITIONAL_EVAL_DUMMY

DIVERSITY_MAX_MOLS = 10000


def diversity_score(smiles: List[Optional[str]], num_decode: int, unconditional: bool=False) -> Tuple[float, float]:
    if unconditional:
        if len(smiles) < DIVERSITY_MAX_MOLS:
            print('Warning: less than ' + str(DIVERSITY_MAX_MOLS) + ' unique smiles found during evaluation. only found ' + str(len(smiles)))
            smiles = smiles + [UNCONDITIONAL_EVAL_DUMMY+str(i) for i in range(DIVERSITY_MAX_MOLS - len(smiles))]
    all_div = []
    for i in range(0, len(smiles), num_decode):
        decoded_smiles = list(set(smiles[i:i + num_decode]))
        if all([s is None for s in decoded_smiles]):  # skip the ones that have no valid translations
            continue
        div = tot = 0
        for j in range(len(decoded_smiles)):
            if decoded_smiles[j] is None:
                continue
            
            for k in range(j + 1, len(decoded_smiles)):
                if decoded_smiles[k] is None:
                    continue

                div += 1 - tanimoto_similarity(decoded_smiles[j], decoded_smiles[k])
                tot += 1
        if tot == 0:
            all_div.append(0.0)
        else:
            all_div.append(div / tot)

    div_mean, div_std = np.mean(all_div), np.std(all_div)
    clear_fp_cache()

    return div_mean, div_std


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_smiles_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with generated smiles')
    args = parser.parse_args()

    smiles, num_decode = get_pred_smiles(args.pred_smiles_dir, return_num_decode=True)

    div_mean, div_std = diversity_score(
        smiles=smiles,
        num_decode=num_decode
    )

    print(f'Diversity = {div_mean} +/- {div_std}')
