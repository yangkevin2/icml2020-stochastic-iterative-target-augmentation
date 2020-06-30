from argparse import ArgumentParser
from typing import List, Optional
import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from chemprop.data.utils import get_smiles
from chemprop.features import get_features_generator

from diversity import diversity_score, DIVERSITY_MAX_MOLS
from novelty import novelty_score
from property import predict_properties
from utils import get_pred_smiles, get_train_smiles, tanimoto_similarity

import sys
sys.path.append('../')
from chemprop_predictor import ChempropPredictor

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

morgan_fingerprint = get_features_generator('morgan')

# Note: a good portion of the evaluation code is copied from a baseline, https://github.com/wengong-jin/iclr19-graph2graph. 

def evaluate_unconditional(pred_smiles_dir: str,
             train_path: str,
             val_path: str,
             checkpoint_dir: Optional[str],
             computed_prop: Optional[str],
             prop_min: float,
             sim_thresholds: List[float],
             chemprop_predictor: ChempropPredictor=None,
             prop_max: float=None):
    pred_smiles, _ = get_pred_smiles(pred_smiles_dir, return_num_decode=True)
    train_smiles = get_train_smiles(train_path)
    print('Num pred smiles:', len(pred_smiles))

    # Get unique pred smiles
    unique_pred_smiles = set(pred_smiles) - {None}
    unique_pred_smiles = list(unique_pred_smiles)
    num_unique_pred_smiles = len(unique_pred_smiles)

    print('Computing property values')
    if chemprop_predictor is not None:
        property_predictions = chemprop_predictor(unique_pred_smiles)
    else:
        property_predictions = predict_properties(unique_pred_smiles, checkpoint_dir, computed_prop)
    property_values = {pred_smile: property_prediction for pred_smile, property_prediction in zip(unique_pred_smiles, property_predictions)}

    print('Filtering by property values')
    pred_smiles = [pred_smile if pred_smile is not None and property_values[pred_smile] is not None and property_values[pred_smile] >= prop_min \
                   and (prop_max is None or property_values[pred_smile] <= prop_max) else None
                   for pred_smile in pred_smiles]
    
    # Get unique pred smiles
    filtered_num_unique_pred_smiles = len(set(pred_smiles) - {None})

    if filtered_num_unique_pred_smiles == 0:
        print(f'Valid molecules above prop threshold = 0% (0/{num_unique_pred_smiles})')
        print('Cannot compute any other metrics with 0 molecules')
        return
    
    print(f'Valid molecules above prop threshold = '
          f'{100 * filtered_num_unique_pred_smiles / num_unique_pred_smiles:.2f}% '
          f'({filtered_num_unique_pred_smiles}/{num_unique_pred_smiles})')
    num_unique_pred_smiles = filtered_num_unique_pred_smiles

    filtered_pred_smiles = pred_smiles
    num_unique_filtered_pred_smiles = len(set(filtered_pred_smiles) - {None})

    print('success num/denom:', sum([1 for x in filtered_pred_smiles if x is not None]), len(pred_smiles))
    paper_success = sum([1 for x in filtered_pred_smiles if x is not None]) / len(pred_smiles) # frac of attempts that succeeded
    filtered_pred_smiles_10k = deepcopy(filtered_pred_smiles)
    filtered_pred_smiles_10k = list(set(filtered_pred_smiles_10k) - {None})
    random.shuffle(filtered_pred_smiles_10k)
    filtered_pred_smiles_10k = filtered_pred_smiles_10k[:DIVERSITY_MAX_MOLS]
    diversity_mean, diversity_std = diversity_score(list(filtered_pred_smiles_10k), DIVERSITY_MAX_MOLS, unconditional=True)
    novelty = novelty_score(filtered_pred_smiles, train_smiles)
    filtered_properties = [property_values[pred_smile] for pred_smile in filtered_pred_smiles if pred_smile is not None]
    property_mean, property_std = np.mean(filtered_properties), np.std(filtered_properties)

    # Print
    print(str(num_unique_filtered_pred_smiles) + ' unique successful translations found in ' + str(len(pred_smiles)) + ' total attempts')
    print(f'Success = {paper_success}')
    print(f'Diversity = {diversity_mean} +/- {diversity_std}')
    print(f'Novelty = {novelty}') # unused for unconditional version
    print(f'Property = {property_mean} +/- {property_std}')


def evaluate(pred_smiles_dir: str,
             train_path: str,
             val_path: str,
             checkpoint_dir: Optional[str],
             computed_prop: Optional[str],
             prop_min: float,
             sim_thresholds: List[float],
             chemprop_predictor: ChempropPredictor=None,
             prop_max: float=None,
             unconditional: bool=False):
    if unconditional:
        evaluate_unconditional(pred_smiles_dir, train_path, val_path, checkpoint_dir, computed_prop, prop_min, sim_thresholds, chemprop_predictor, prop_max)
        return
    # Get smiles
    pred_smiles, num_decode = get_pred_smiles(pred_smiles_dir, return_num_decode=True)
    train_smiles = get_train_smiles(train_path)
    source_smiles = get_smiles(val_path, header=False)

    assert len(source_smiles) * num_decode == len(pred_smiles)

    # Get unique pred smiles
    unique_pred_smiles = set(pred_smiles) - {None}
    unique_pred_smiles = list(unique_pred_smiles)
    num_unique_pred_smiles = len(unique_pred_smiles)

    print('Computing tanimoto similarities from pred molecule to source molecule')
    tanimoto_similarities = {}
    for i, pred_smile in tqdm(enumerate(pred_smiles), total=len(pred_smiles)):
        source_smile = source_smiles[i // num_decode]
        tanimoto_similarities[(source_smile, pred_smile)] = tanimoto_similarity(source_smile, pred_smile) if pred_smile is not None else None

    print('Computing property values')
    if chemprop_predictor is not None:
        property_predictions = chemprop_predictor(unique_pred_smiles)
    else:
        property_predictions = predict_properties(unique_pred_smiles, checkpoint_dir, computed_prop)
    property_values = {pred_smile: property_prediction for pred_smile, property_prediction in zip(unique_pred_smiles, property_predictions)}

    print('Filtering by property values')
    pred_smiles = [pred_smile if pred_smile is not None and property_values[pred_smile] is not None and property_values[pred_smile] >= prop_min \
                   and (prop_max is None or property_values[pred_smile] <= prop_max) else None
                   for pred_smile in pred_smiles]

    # Get unique pred smiles
    filtered_num_unique_pred_smiles = len(set(pred_smiles) - {None})

    if filtered_num_unique_pred_smiles == 0:
        print(f'Valid molecules above prop threshold = 0% (0/{num_unique_pred_smiles})')
        print('Cannot compute any other metrics with 0 molecules')
        return

    print(f'Valid molecules above prop threshold = '
          f'{100 * filtered_num_unique_pred_smiles / num_unique_pred_smiles:.2f}% '
          f'({filtered_num_unique_pred_smiles}/{num_unique_pred_smiles})')
    num_unique_pred_smiles = filtered_num_unique_pred_smiles

    for sim_threshold in sim_thresholds:
        print(f'Minimum tanimoto similarity to source molecule allowed = {sim_threshold}')

        # Filtering by tanimoto similarity
        filtered_pred_smiles = []
        for i, pred_smile in enumerate(pred_smiles):
            source_smile = source_smiles[i // num_decode]
            filtered_pred_smiles.append(pred_smile if pred_smile is not None \
                        and tanimoto_similarities[(source_smile, pred_smile)] is not None \
                        and tanimoto_similarities[(source_smile, pred_smile)] >= sim_threshold else None)
        num_unique_filtered_pred_smiles = len(set(filtered_pred_smiles) - {None})

        print(f'Percent of unique pred smiles after filtering by tanimoto = '
              f'{100 * num_unique_filtered_pred_smiles / num_unique_pred_smiles:.2f}% '
              f'({num_unique_filtered_pred_smiles}/{num_unique_pred_smiles})')

        if num_unique_filtered_pred_smiles == 0:
            print('No molecules remaining, skipping')
            continue

        # Evaluate
        succeeded_val, tot_val = 0, 0
        for i in range(0, len(filtered_pred_smiles), num_decode):
            decoded_smiles = filtered_pred_smiles[i:i+num_decode]
            if any(s is not None for s in decoded_smiles):
                succeeded_val += 1
            tot_val += 1

        paper_success = succeeded_val / tot_val
        diversity_mean, diversity_std = diversity_score(filtered_pred_smiles, num_decode)
        novelty = novelty_score(filtered_pred_smiles, train_smiles)
        filtered_properties = [property_values[pred_smile] for pred_smile in filtered_pred_smiles if pred_smile is not None]
        property_mean, property_std = np.mean(filtered_properties), np.std(filtered_properties)

        print(f'Success = {paper_success}')
        print(f'Diversity = {diversity_mean} +/- {diversity_std}')
        print(f'Novelty = {novelty}')
        print(f'Property = {property_mean} +/- {property_std}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_smiles_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with generated smiles')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to a file containing .txt files with train source/target smiles')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to a file containing .txt files with validation source smiles')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Path to directory containing chemprop model .pt files')
    parser.add_argument('--computed_prop', type=str, choices=['penalized_logp', 'logp', 'qed', 'sascore', 'drd2'], default=None,
                        help='Computed property to evaluate')
    parser.add_argument('--prop_min', type=float, default=-float('inf'),
                        help='Minimum property value for a predicted molecule to be included in the evaluation')
    parser.add_argument('--prop_max', type=float, default=None, help='max property of translation to be considered successful')
    parser.add_argument('--sim_thresholds', type=float, nargs='+', default=[1.0, 0.8, 0.6, 0.4, 0.2],
                        help='Maximum tanimoto similarity to source molecule allowed')
    parser.add_argument('--unconditional', action='store_true', default=False, help='unconditional setting')
    args = parser.parse_args()

    assert args.checkpoint_dir is not None or args.computed_prop is not None

    evaluate(
        pred_smiles_dir=args.pred_smiles_dir,
        train_path=args.train_path,
        val_path=args.val_path,
        checkpoint_dir=args.checkpoint_dir,
        computed_prop=args.computed_prop,
        prop_min=args.prop_min,
        sim_thresholds=args.sim_thresholds,
        prop_max=args.prop_max,
        unconditional=args.unconditional
    )
