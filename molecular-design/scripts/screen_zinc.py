from argparse import ArgumentParser
import os

import sys
sys.path.append('../')
from chemprop_predictor import ChempropPredictor

def screen(args):
    args.batch_size = 64
    args.prop_index, args.neg_threshold, args.features_generator, args.chemprop_dir = 0, None, None, None
    computed_predictor = ChempropPredictor(args)

    with open(args.save_path, 'w') as wf:
        for _, _, files in os.walk(args.screen_dir):
            for filename in files:
                with open(os.path.join(args.screen_dir, filename), 'r') as rf:
                    rf.readline()
                    smiles = [line.strip() for line in rf]
                    computed_preds = computed_predictor(smiles)
                    for i in range(len(smiles)):
                        if computed_preds[i] > args.prop_min and computed_preds[i] < args.prop_max:
                            wf.write(smiles[i] + '\n')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prop_min', type=float, default=-float('inf'), help='min property of precursor')
    parser.add_argument('--prop_max', type=float, default=None, help='max property of precursor')
    parser.add_argument('--computed_prop', type=str, choices=['penalized_logp', 'logp', 'qed', 'sascore', 'drd2'], default=None,
                        help='Computed property to evaluate')
    parser.add_argument('--screen_dir', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    screen(args)