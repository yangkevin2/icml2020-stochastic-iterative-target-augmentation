from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import numpy as np

from chemprop.parsing import add_predict_args, update_checkpoint_args
from chemprop.train.make_predictions import make_predictions

from utils import get_pred_smiles
from props.properties import penalized_logp, logp, qed, sascore, drd2


def predict_properties(pred_smiles: List[Optional[str]], checkpoint_dir: str, computed_prop: str) -> List[float]:
    # Check that exactly one of checkpoint_dir and computed_prop is provided
    assert (checkpoint_dir is None) != (computed_prop is None)

    pred_smiles = [smiles for smiles in pred_smiles if smiles is not None]

    # Create and modify predict args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args([])

    if checkpoint_dir is not None:
        args.test_path = 'None'
        args.checkpoint_dir = checkpoint_dir
        update_checkpoint_args(args)
        args.quiet = True

        print('Make predictions')
        with NamedTemporaryFile() as temp_file:
            args.preds_path = temp_file.name
            property_predictions = make_predictions(args, smiles=pred_smiles)

        property_predictions = [property_prediction[0] for property_prediction in property_predictions]
    else:
        if computed_prop == 'penalized_logp':
            scorer = penalized_logp
        elif computed_prop == 'logp':
            scorer = logp
        elif computed_prop == 'qed':
            scorer = qed
        elif computed_prop == 'sascore':
            scorer = sascore
        elif computed_prop == 'drd2':
            scorer = drd2
        else:
            raise ValueError(f'Computed property "{computed_prop}" not supported')

        property_predictions = [scorer(s) for s in pred_smiles]

    return property_predictions


def property_score(pred_smiles: List[Optional[str]], checkpoint_dir: str, computed_prop: str) -> Tuple[float, float]:
    property_predictions = predict_properties(pred_smiles, checkpoint_dir, computed_prop)

    return np.mean(property_predictions), np.std(property_predictions)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_smiles_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with generated smiles')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Path to directory containing chemprop model .pt files')
    parser.add_argument('--computed_prop', type=str, choices=['penalized_logp', 'logp', 'qed', 'sascore', 'drd2'], default=None,
                        help='Computed property to evaluate')
    args = parser.parse_args()

    pred_smiles = get_pred_smiles(args.pred_smiles_dir)

    property_mean, property_std = property_score(
        pred_smiles=pred_smiles,
        checkpoint_dir=args.checkpoint_dir,
        computed_prop=args.computed_prop
    )

    print(f'Property = {property_mean} +/- {property_std}')
