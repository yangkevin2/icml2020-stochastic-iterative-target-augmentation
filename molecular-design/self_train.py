from argparse import Namespace
from copy import deepcopy
from typing import List, Optional, Tuple

import math
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
from tqdm import tqdm
from scipy.spatial.distance import cosine as cosine_distance
import numpy as np
import torch.nn as nn
import random

from chemprop_predictor import ChempropPredictor
from dataset import PairDataset
from model import Model
from util import pad_batch

def evaluate_translation(args, 
                         src_smile,
                         translation,
                         prop, 
                         prop_min, 
                         prop_max, 
                         sim_threshold,
                         i=None,
                         no_filter=False,
                         unconditional=False):
    if no_filter:
        return True
    if prop is None or prop <= prop_min:
        return False
    if prop_max is not None and prop > prop_max:
        return False
    if unconditional:
        return True # no need to check the similarity

    # Convert to mol and then to morgan
    src_mol, translation_mol = Chem.MolFromSmiles(src_smile), Chem.MolFromSmiles(translation)

    if src_mol is None or translation_mol is None:
        return False

    src_morgan = AllChem.GetMorganFingerprintAsBitVect(src_mol, 2, nBits=2048, useChirality=False)
    translation_morgan = AllChem.GetMorganFingerprintAsBitVect(translation_mol, 2, nBits=2048, useChirality=False)

    # Check if similarity is below threshold
    if DataStructs.TanimotoSimilarity(src_morgan, translation_morgan) < sim_threshold:
        return False
    return True


def find_good_translations(src_smiles: List[str],
                           translations: List[str],
                           args: Namespace,
                           props: List[Optional[float]],
                           tgt_props: List[Optional[float]],
                           prop_min: float,
                           prop_max: float,
                           sim_threshold: float,
                           translation_index_batches = None):
    good_translation_indices = []
    for i, (src_smile, translation, prop, tgt_prop) in enumerate(zip(src_smiles, translations, props, tgt_props)):
        if evaluate_translation(args=args,
                                src_smile=src_smile,
                                translation=translation,
                                prop=prop,
                                prop_min=prop_min,
                                prop_max=prop_max,
                                sim_threshold=sim_threshold,
                                i=i,
                                no_filter=args.no_filter):
            good_translation_indices.append(i)

    return good_translation_indices

def generate_self_train_translations(train_dataset: PairDataset,
                                     model: Model,
                                     chemprop_predictor: ChempropPredictor,
                                     args: Namespace,
                                     k: int = 4) -> Tuple[List[List[str]], List[List[float]]]:
    # Copy train dataset since we're going to modify it
    train_dataset = deepcopy(train_dataset)

    # Keep a copy of the original targets and their properties for molecules which don't get enough translations
    src = deepcopy(train_dataset.src)
    tgt = deepcopy(train_dataset.tgt)

    if not (hasattr(train_dataset, 'tgt_props') and train_dataset.tgt_props is not None):
        train_dataset.tgt_props = chemprop_predictor(train_dataset.tgt_smiles)

    tgt_props = deepcopy(train_dataset.tgt_props)

    # Increase batch size since using no_grad
    train_dataset.set_batch_size(train_dataset.batch_size * 4)

    # Initialize mapping from index in train_dataset.src to index in all_translations/props
    # as train_dataset.src gets filtered
    index_mapping = list(range(len(train_dataset)))

    if args.unconditional:
        for batch_src, _ in train_dataset: # get an arbitrarily batch_src as a dummy
            # Pad batch src
            batch_src, lengths_src = pad_batch(batch_src, train_dataset.pad_index)

            # To cuda
            batch_src, lengths_src = batch_src.cuda(), lengths_src.cuda()
            break

        if args.dupe:
            good_translations_and_props = []
        else:
            good_translations_and_props = set()
        for _ in range(args.num_translations_per_input):
            if args.dupe:
                translations = []
            else:
                translations = set()
                translations.update(train_dataset.tgt_smiles)
            for _ in range(math.ceil(len(train_dataset) / train_dataset.batch_size)): # keep same num attempts
                # Predict
                with torch.no_grad():
                    batch_indices = model.predict(batch_src, lengths_src, sample=True)  # seqlen x bs
                
                # Transpose
                batch_indices = torch.stack(batch_indices, dim=1).tolist()  # bs x seqlen

                # Convert predicted indices to smiles
                batch_translations = [train_dataset.indices2smiles(indices) for indices in batch_indices]
                if args.dupe:
                    translations += batch_translations
                else:
                    translations.update(batch_translations)

            if not args.dupe:
                translations = translations.difference(train_dataset.tgt_smiles) # remove the ones we already had
                translations = list(translations)
            props = chemprop_predictor(translations)
            chemprop_predictor.clear_cache()
            for t, prop in zip(translations, props):
                if prop is not None and prop > args.prop_min and (args.prop_max is None or prop <= args.prop_max):
                    if args.dupe:
                        good_translations_and_props.append((t, prop))
                    else:
                        good_translations_and_props.add((t, prop))
            if len(good_translations_and_props) >= k * len(train_dataset):
                break
        good_translations_and_props = list(good_translations_and_props)
        random.shuffle(good_translations_and_props)
        good_translations_and_props = good_translations_and_props[:k * len(train_dataset)]
        return [x[0] for x in good_translations_and_props], [x[1] for x in good_translations_and_props] # translations, props

    # Initialize set of indices which have k translations
    finished_indices = set()

    # Generate translations
    all_translations_and_props = [set() for _ in range(len(train_dataset))]

    for _ in range(args.num_translations_per_input):
        translations = []
        translation_index_batches = []

        for batch_src, _ in tqdm(train_dataset, total=math.ceil(len(train_dataset) / train_dataset.batch_size)):
            # Pad batch src
            batch_src, lengths_src = pad_batch(batch_src, train_dataset.pad_index)

            # To cuda
            batch_src, lengths_src = batch_src.cuda(), lengths_src.cuda()

            # Predict
            with torch.no_grad():
                batch_indices = model.predict(batch_src, lengths_src, sample=True)  # seqlen x bs

            # Transpose
            batch_indices = torch.stack(batch_indices, dim=1).tolist()  # bs x seqlen

            # Convert predicted indices to smiles
            batch_translations = [train_dataset.indices2smiles(indices) for indices in batch_indices]

            # Extend
            translations += batch_translations
            translation_index_batches.append(batch_indices)

        # Predict properties
        props = chemprop_predictor(translations)
        chemprop_predictor.clear_cache()
        assert len(train_dataset) == len(translations) == len(props)

        # Find indices of good translations; if source side, we're checking tgt smiles with the new translations for similarity
        good_translation_indices = find_good_translations(
            src_smiles=train_dataset.src_smiles,
            translations=translations,
            args=args,
            props=props,
            tgt_props=tgt_props,
            prop_min=args.prop_min,
            prop_max=args.prop_max,
            sim_threshold=args.morgan_similarity_threshold,
            translation_index_batches=translation_index_batches
        )

        # Add good translations to all translations
        for i in good_translation_indices:
            j = index_mapping[i]  # convert from index in current train_dataset.src to index in all_translations/props

            # Add good translation
            all_translations_and_props[j].add((translations[i], props[i]))

            # Indicate if we have enough translations for a molecule
            if len(all_translations_and_props[j]) >= k:
                finished_indices.add(j)

        # Update index_mapping to only keep src/tgt molecules without enough good translations
        keep_indices = [i for i in range(len(translations)) if index_mapping[i] not in finished_indices]

        train_dataset.src = [train_dataset.src[i] for i in keep_indices]
        train_dataset.src_smiles = [train_dataset.src_smiles[i] for i in keep_indices]
        train_dataset.tgt = [train_dataset.tgt[i] for i in keep_indices]
        train_dataset.tgt_smiles = [train_dataset.tgt_smiles[i] for i in keep_indices]
        train_dataset.tgt_props = [train_dataset.tgt_props[i] for i in keep_indices]

        index_mapping = [index_mapping[i] for i in keep_indices]

    # Check lengths
    assert len(all_translations_and_props) == len(tgt) == len(tgt_props)

    # Add actual target to any molecules which did not get enough translations
    all_translations, all_props = [], []
    for i in range(len(all_translations_and_props)):
        all_translations_and_props[i] = list(all_translations_and_props[i])
        all_translations_and_props[i] = [(train_dataset.smiles2indices(translation), prop) for translation, prop in all_translations_and_props[i]]

        if len(all_translations_and_props[i]) < k:
            all_translations_and_props[i] += [(tgt[i], tgt_props[i])] * (k - len(all_translations_and_props[i]))

        all_translations.append([pair[0] for pair in all_translations_and_props[i]])
        all_props.append([pair[1] for pair in all_translations_and_props[i]])

    return all_translations, all_props
