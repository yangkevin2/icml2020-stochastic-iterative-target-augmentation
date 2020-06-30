from argparse import ArgumentParser
from typing import List, Optional

from utils import get_pred_smiles, get_train_smiles


def novelty_score(pred_smiles: List[Optional[str]], train_smiles: List[str]) -> float:
    pred_smiles = [smiles for smiles in pred_smiles if smiles is not None]
    pred_smiles, train_smiles = set(pred_smiles), set(train_smiles)
    intersection = pred_smiles & train_smiles
    novelty = 1 - len(intersection) / len(train_smiles)

    return novelty


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_smiles_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with generated smiles')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to a file containing .txt files with source/target smiles')
    args = parser.parse_args()

    pred_smiles = get_pred_smiles(args.pred_smiles_dir)
    train_smiles = get_train_smiles(args.train_path)

    novelty = novelty_score(
        pred_smiles=pred_smiles,
        train_smiles=train_smiles
    )

    print(f'Novelty = {novelty}')
