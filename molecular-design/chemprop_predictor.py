from argparse import Namespace
from multiprocessing import Pool
from typing import List, Optional
import os
from tqdm import tqdm
from collections import defaultdict

import numpy as np

from chemprop.train.predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.features import get_features_generator, clear_cache

from dataset import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN
from eval.props.properties import penalized_logp, logp, qed, sascore, drd2
from util import pad_batch

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

class ChempropPredictor:
    def __init__(self, args: Namespace):

        if args.computed_prop is not None:
            self.computed_prop = True

            if args.computed_prop == 'penalized_logp':
                self.scorer = penalized_logp
            elif args.computed_prop == 'logp':
                self.scorer = logp
            elif args.computed_prop == 'qed':
                self.scorer = qed
            elif args.computed_prop == 'sascore':
                self.scorer = sascore
            elif args.computed_prop == 'drd2':
                self.scorer = drd2
            else:
                raise ValueError
            return

        self.computed_prop = False

        chemprop_paths = []
        for root, _, files in os.walk(args.chemprop_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    chemprop_paths.append(os.path.join(root, fname))

        self.scaler, self.features_scaler = load_scalers(chemprop_paths[0])
        self.train_args = load_args(chemprop_paths[0])
        if self.train_args.features_path is not None:
            self.train_args.features_path = None
            self.train_args.features_generator = ['rdkit_2d_normalized'] # just assume this
        self.num_tasks = self.train_args.num_tasks
        self.batch_size = args.batch_size * 4
        self.features_generator = get_features_generator(args.features_generator[0]) if args.features_generator is not None else None
        self.neg_threshold = args.neg_threshold
        self.prop_index = args.prop_index

        self.chemprop_models = []
        for checkpoint_path in chemprop_paths:
            self.chemprop_models.append(load_checkpoint(checkpoint_path, cuda=True))
    
    def generate_features(self, data): # note - this isn't used in our QED or DRD2 experiments
        all_mols = list(d.mol for d in data)

        valid_indices = [i for i in range(len(all_mols)) if all_mols[i] is not None]
        mols = [mol for mol in all_mols if mol is not None]

        features = []
        for mol in mols:
            try:
                features.append(self.features_generator(mol))
            except:
                features.append(None) # rare crashes in feature generators

        assert len(valid_indices) == len(features)

        for i, valid_index in enumerate(valid_indices):
            data[valid_index].features = features[i]
    
    def clear_cache(self):
        clear_cache()

    def __call__(self, smiles: List[Optional[str]] = None, smiles2: List[Optional[str]] = None) -> List[Optional[List[float]]]:
        if self.computed_prop:
            # Identity non-None smiles
            if len(smiles) > 0:
                valid_indices, valid_smiles = zip(*[(i, smile) for i, smile in enumerate(smiles) if smile is not None])
            else:
                valid_indices, valid_smiles = [], []
            
            valid_props = [self.scorer(valid_smile) for valid_smile in valid_smiles]

            # Combine properties of non-None smiles with Nones
            props = [None] * len(smiles)
            for i, prop in zip(valid_indices, valid_props):
                props[i] = prop

            return props
            
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])
        if self.features_generator is not None: 
            self.generate_features(test_data)
            valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None and test_data[i].features is not None]
            test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        # Edge case if empty list of smiles is provided
        if len(test_data) == 0:
            return [None] * len(full_data)

        # Normalize features
        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)

        # Predict with each model individually and sum predictions
        sum_preds = np.zeros((len(test_data), self.num_tasks))
        for chemprop_model in self.chemprop_models:
            model_preds = predict(
                model=chemprop_model,
                data=test_data,
                batch_size=self.batch_size,
                scaler=self.scaler
            )
            sum_preds += np.array(model_preds)

        # Ensemble predictions
        avg_preds = sum_preds / len(self.chemprop_models)
        avg_preds = avg_preds.tolist()

        # Put Nones for invalid smiles
        full_preds = [None] * len(full_data)
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]

        if self.neg_threshold:
            return [-p[self.prop_index] if p is not None else None for p in full_preds]
        else:
            return [p[self.prop_index] if p is not None else None for p in full_preds]
