from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
import math

import sys
sys.path.append('../')
from chemprop_predictor import ChempropPredictor

def compare(args):
    args.batch_size = 64
    args.prop_index, args.neg_threshold, args.features_generator = 0, None, None
    computed_predictor = ChempropPredictor(args)
    args.computed_prop = None
    chemprop_predictor = ChempropPredictor(args)
    with open(args.test_path, 'r') as rf:
        smiles = [line.strip() for line in rf]
    computed_preds = computed_predictor(smiles)
    chemprop_preds = chemprop_predictor(smiles)
    error = math.sqrt(mean_squared_error(computed_preds, chemprop_preds))
    print('rmse', error)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_path', type=str, default='../data/drd2/test.txt')
    parser.add_argument('--chemprop_dir', type=str, default=None)
    parser.add_argument('--computed_prop', type=str, choices=['penalized_logp', 'logp', 'qed', 'sascore', 'drd2'], default=None,
                        help='Computed property to evaluate')
    
    args = parser.parse_args()
    compare(args)