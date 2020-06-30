from argparse import ArgumentParser, Namespace
import math
import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm, trange

from chemprop.features import get_available_features_generators

from chemprop_predictor import ChempropPredictor
from dataset import PairDataset, SourceDataset
from model import Model
from self_train import generate_self_train_translations, evaluate_translation
from util import pad_batch, LossFunction, save_model, load_model, compute_grad_norm, parameters_to_vector

sys.path.append('eval')
from eval.evaluate import evaluate


def train(model: Model,
          train_dataset: PairDataset,
          criterion: LossFunction,
          optimizer: optim.Optimizer,
          max_grad_norm=None,
          original_parameter_vector=None,
          parameter_crit=None,
          parameter_crit_weight=None):
    assert (original_parameter_vector is None) == (parameter_crit is None) == (parameter_crit_weight is None)
    model.train()
    print_every = 200
    count = 0
    total_loss = 0
    total_param_loss = 0
    warned = False

    for batch_src, batch_tgt in tqdm(train_dataset, total=math.ceil(len(train_dataset) / train_dataset.batch_size)):
        optimizer.zero_grad()

        batch_src, lengths_src = pad_batch(batch_src, train_dataset.pad_index)
        batch_tgt, lengths_tgt = pad_batch(batch_tgt, train_dataset.pad_index)
        batch_src, lengths_src, batch_tgt, lengths_tgt = batch_src.cuda(), lengths_src.cuda(), batch_tgt.cuda(), lengths_tgt.cuda()

        output, mu, logvar = model(batch_src, lengths_src, batch_tgt, lengths_tgt)

        prediction_padding = torch.LongTensor([train_dataset.pad_index for _ in range(lengths_tgt.size(0))]).unsqueeze(0).cuda()  # 1 x bs
        prediction_tgt = torch.cat([batch_tgt[1:, :], prediction_padding], dim=0)  # still seqlen x bs

        loss = criterion.forward(output, mu, logvar, prediction_tgt)
        total_loss += loss
        if original_parameter_vector is not None:
            parameter_diff_loss = parameter_crit(parameters_to_vector(model.parameters()), original_parameter_vector)
            total_param_loss += parameter_diff_loss
            loss = loss + parameter_diff_loss * parameter_crit_weight
        loss.backward()
        if max_grad_norm is not None:
            total_norm = compute_grad_norm(model)
            if not warned and total_norm > max_grad_norm:
                print('clipping gradient norm')
                warned = True
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        count += 1
        if count % print_every == 0: 
            print('average train loss: {}'.format(total_loss/print_every))
            total_loss = 0
            if original_parameter_vector is not None:
                print('average train param diff loss: {}'.format(total_param_loss/print_every))
                total_param_loss = 0


def validate(model: Model,
             val_dataset: PairDataset,
             criterion: LossFunction):
    model.eval()
    total_loss = 0
    count = 0

    for batch_src, batch_tgt in tqdm(val_dataset, total=math.ceil(len(val_dataset)/val_dataset.batch_size)):
        batch_src, lengths_src = pad_batch(batch_src, val_dataset.pad_index)
        batch_tgt, lengths_tgt = pad_batch(batch_tgt, val_dataset.pad_index)
        batch_src, lengths_src, batch_tgt, lengths_tgt = batch_src.cuda(), lengths_src.cuda(), batch_tgt.cuda(), lengths_tgt.cuda()

        with torch.no_grad():
            output, mu, logvar = model(batch_src, lengths_src, batch_tgt, lengths_tgt, use_vae=False)

        prediction_padding = torch.LongTensor([val_dataset.pad_index for _ in range(lengths_tgt.size(0))]).unsqueeze(0).cuda()  # 1 x bs
        prediction_tgt = torch.cat([batch_tgt[1:, :], prediction_padding], dim=0)  # still seqlen x bs
        loss = criterion.forward(output, mu, logvar, prediction_tgt)
        total_loss += loss
        count += 1

    average_val_loss = total_loss / count
    print('average val loss: {}'.format(average_val_loss))
    return average_val_loss


def predict(model: Model,
            predict_dataset: SourceDataset,
            save_dir: str,
            args: Namespace,
            chemprop_predictor: ChempropPredictor = None,
            sample: bool = False,
            num_predictions: int = 10,
            print_filter_frac: bool = False):
    model.eval()
    passed_filter = 0
    for i in range(1 if args.unconditional else num_predictions):
        with open(os.path.join(save_dir, 'prediction_' + str(i) + '.txt'), 'w') as f:
            all_smiles = [[] for _ in range(args.max_translations_per_pred)]
            if args.unconditional:
                batch_src, lengths_src = torch.zeros(predict_dataset.batch_size),\
                                         torch.zeros(predict_dataset.batch_size)
                for _ in range(math.ceil(num_predictions / predict_dataset.batch_size)):
                    for k in range(args.max_translations_per_pred if chemprop_predictor is not None else 1):

                        with torch.no_grad():
                            output = model.predict(batch_src, lengths_src, sample=sample)  # list of tensors of shape bs
                        
                        output = [o.tolist() for o in output]
                        attempt_smiles = []
                        for j in range(lengths_src.size(0)):
                            tokens = [output[k][j] for k in range(len(output))]
                            smiles = predict_dataset.indices2smiles(tokens)
                            attempt_smiles.append(smiles)
                        all_smiles[k] += attempt_smiles
                for k in range(args.max_translations_per_pred):
                    all_smiles[k] = all_smiles[k][:num_predictions]
                # write smiles to file
                for j in range(len(all_smiles[0])):
                    smiles = all_smiles[0][j]
                    if chemprop_predictor is not None:
                        preds = chemprop_predictor([all_smiles[k][j] for k in range(args.max_translations_per_pred)])
                        for k in range(args.max_translations_per_pred):
                            if evaluate_translation(args=args, 
                                                    src_smile=None,
                                                    translation=all_smiles[k][j],
                                                    prop=preds[k],
                                                    prop_min=args.prop_min,
                                                    prop_max=args.prop_max,
                                                    sim_threshold=args.morgan_similarity_threshold,
                                                    no_filter=args.no_filter,
                                                    unconditional=True): 
                                smiles = all_smiles[k][j]
                                if k == 0:
                                    passed_filter += 1 # sample of 1 try per validation/test set molecule
                                break
                    f.write(''.join(smiles))
                    f.write('\n')
            else:
                for batch_src in tqdm(predict_dataset, total=math.ceil(len(predict_dataset)/predict_dataset.batch_size)):
                    batch_smiles = [predict_dataset.indices2smiles(src) for src in batch_src]
                    all_smiles = []

                    batch_src, lengths_src = pad_batch(batch_src, predict_dataset.pad_index)
                    batch_src, lengths_src = batch_src.cuda(), lengths_src.cuda()
                    for _ in range(args.max_translations_per_pred if chemprop_predictor is not None else 1):

                        with torch.no_grad():
                            output = model.predict(batch_src, lengths_src, sample=sample)  # list of tensors of shape bs
                        
                        output = [o.tolist() for o in output]
                        attempt_smiles = []
                        for j in range(lengths_src.size(0)):
                            tokens = [output[k][j] for k in range(len(output))]
                            smiles = predict_dataset.indices2smiles(tokens)
                            attempt_smiles.append(smiles)
                        all_smiles.append(attempt_smiles)
                    # write smiles to file
                    for j in range(lengths_src.size(0)):
                        smiles = all_smiles[0][j]
                        if chemprop_predictor is not None:
                            preds = chemprop_predictor([all_smiles[k][j] for k in range(args.max_translations_per_pred)])
                            for k in range(args.max_translations_per_pred):
                                if evaluate_translation(args=args, 
                                                        src_smile=batch_smiles[j],
                                                        translation=all_smiles[k][j],
                                                        prop=preds[k],
                                                        prop_min=args.prop_min,
                                                        prop_max=args.prop_max,
                                                        sim_threshold=args.morgan_similarity_threshold,
                                                        no_filter=args.no_filter): 
                                    smiles = all_smiles[k][j]
                                    if k == 0:
                                        passed_filter += 1 # sample of 1 try per validation/test set molecule
                                    break
                        f.write(''.join(smiles))
                        f.write('\n')
    passed_filter_frac = passed_filter / (num_predictions * len(predict_dataset))
    if print_filter_frac:
        print('fraction of first attempts passing external filter: ' + str(passed_filter_frac))


def main(args: Namespace):
    if args.unconditional:
        assert args.morgan_similarity_threshold == 0 # shouldn't care about inputs in this case

    i2s = None

    if args.checkpoint_dir is not None:
        assert args.checkpoint_path is None
        for _, _, files in os.walk(args.checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    args.checkpoint_path = os.path.join(args.checkpoint_dir, fname)

    if args.checkpoint_path is not None: 
        print('loading model from checkpoint')
        model, i2s = load_model(args)

    full_train_dataset = PairDataset(
        path=args.train_path,
        i2s=i2s,
        batch_size=args.batch_size,
        extra_vocab_path=args.extra_precursors_path if args.extra_precursors_path is not None else None,
        max_data=args.train_max_data if args.train_max_data is not None else None
    )
    pair_datasets = full_train_dataset.split([0.9, 0.1], seed=0)
    train_dataset, val_dataset = pair_datasets[0], pair_datasets[1]
    predict_dataset = SourceDataset(
        path=args.val_path,
        i2s=train_dataset.i2s,
        s2i=train_dataset.s2i,
        pad_index=train_dataset.pad_index,
        start_index=train_dataset.start_index,
        end_index=train_dataset.end_index,
        batch_size=args.batch_size)

    if args.checkpoint_path is None:
        print('building model from scratch')
        model = Model(
            args=args,
            vocab_size=len(train_dataset.i2s),
            pad_index=train_dataset.pad_index,
            start_index=train_dataset.start_index,
            end_index=train_dataset.end_index
        )
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    print(model)
    print('num params: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = model.cuda()

    chemprop_predictor = ChempropPredictor(args)

    criterion = LossFunction(train_dataset.pad_index, args.kl_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    for epoch in range(args.epochs):
        print('epoch {}'.format(epoch))
        train_dataset.reshuffle(seed=epoch)
        train(
            model=model,
            train_dataset=train_dataset,
            criterion=criterion,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm
        )
        val_loss = validate(
            model=model,
            val_dataset=val_dataset,
            criterion=criterion
        )
        os.makedirs(os.path.join(args.save_dir, 'epoch' + str(epoch)), exist_ok=True)
        train_dataset.save(os.path.join(args.save_dir, 'epoch' + str(epoch), 'train_pairs.csv'))
        save_model(
            model=model,
            i2s=train_dataset.i2s,
            path=os.path.join(args.save_dir, 'epoch' + str(epoch), 'val_loss_{}.pt'.format(val_loss))
        )
        predict(
            model=model,
            predict_dataset=predict_dataset,
            save_dir=os.path.join(args.save_dir, 'epoch' + str(epoch)),
            args=args,
            chemprop_predictor=chemprop_predictor if not args.no_predictor_at_val else None,
            sample=not args.greedy_prediction,
            num_predictions=args.val_num_predictions,
            print_filter_frac=args.print_filter_frac
        )
        if epoch % args.evaluate_every == 0:
            evaluate(
                pred_smiles_dir=os.path.join(args.save_dir, 'epoch' + str(epoch)),
                train_path=args.train_path,
                val_path=args.val_path,
                checkpoint_dir=args.chemprop_dir,
                computed_prop=args.computed_prop,
                prop_min=args.prop_min,
                sim_thresholds=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                chemprop_predictor=chemprop_predictor,
                prop_max=args.prop_max,
                unconditional=args.unconditional
            )
        scheduler.step()
    
    if args.self_train_epochs > 0:
        # store parameters of current model for a loss to constrain it not to stray too far
        original_parameter_vector = parameters_to_vector(model.parameters()).data
        parameter_crit = nn.MSELoss()

        args.epoch_length = len(train_dataset.src) // 2

        # Get properties of target molecules in train set
        train_dataset.tgt_props = np.array(chemprop_predictor(train_dataset.tgt_smiles))
        
        augmented_train_dataset = deepcopy(train_dataset)

        epochs_to_dataset_creation = 0
        for epoch in range(args.epochs, args.epochs + args.self_train_epochs):
            print('self train epoch {}'.format(epoch))

            if epochs_to_dataset_creation == 0:
                train_dataset.reshuffle(seed=epoch)
                if args.self_train_max_data is not None:
                    self_train_dataset = deepcopy(train_dataset)
                    self_train_dataset.src, self_train_dataset.tgt = \
                            self_train_dataset.src[:args.self_train_max_data], self_train_dataset.tgt[:args.self_train_max_data] 
                    self_train_dataset.src_smiles, self_train_dataset.tgt_smiles = \
                            self_train_dataset.src_smiles[:args.self_train_max_data], self_train_dataset.tgt_smiles[:args.self_train_max_data]
                    if hasattr(self_train_dataset, 'src_props'):
                        self_train_dataset.src_props = self_train_dataset.src_props[:args.self_train_max_data]
                    if hasattr(self_train_dataset, 'tgt_props'):
                        self_train_dataset.tgt_props = self_train_dataset.tgt_props[:args.self_train_max_data]
                else:
                    self_train_dataset = deepcopy(train_dataset)
                if args.extra_precursors_path is not None:
                    self_train_dataset.add_dummy_pairs(args.extra_precursors_path)
                translations, props = generate_self_train_translations(
                    train_dataset=self_train_dataset,
                    model=model,
                    chemprop_predictor=chemprop_predictor,
                    args=args,
                    k=args.k
                )

                if not args.keep_translations: # drop old translations and restart
                    augmented_train_dataset = deepcopy(self_train_dataset)

                if args.unconditional:
                    new_train_dataset = deepcopy(self_train_dataset)
                    new_train_dataset.tgt_smiles = translations
                    new_train_dataset.tgt = [list(self_train_dataset.smiles2indices(smiles)) for smiles in new_train_dataset.tgt_smiles]
                    new_train_dataset.tgt = np.array(new_train_dataset.tgt)
                    new_train_dataset.src_smiles = translations # any dummy is fine
                    new_train_dataset.src = [list(self_train_dataset.smiles2indices(smiles)) for smiles in new_train_dataset.src_smiles]
                    new_train_dataset.src = np.array(new_train_dataset.src)
                else:
                    new_train_dataset = deepcopy(self_train_dataset)
                    new_train_dataset.src = np.concatenate([self_train_dataset.src for _ in range(args.k)])
                    new_train_dataset.src_smiles = []
                    for _ in range(args.k):
                        new_train_dataset.src_smiles += self_train_dataset.src_smiles
                    new_train_dataset.tgt = []
                    for i in range(args.k):
                        new_train_dataset.tgt += [translations[j][i] for j in range(len(translations))]
                    new_train_dataset.tgt_smiles = [self_train_dataset.indices2smiles(indices) for indices in new_train_dataset.tgt]
                    new_train_dataset.tgt = np.array(new_train_dataset.tgt)
                if args.replace_old_dataset:
                    augmented_train_dataset = new_train_dataset
                else:
                    augmented_train_dataset.add(new_train_dataset)
                
                if not args.unconditional:
                    augmented_train_dataset.filter_dummy_pairs(need_props=False) # filters src == tgt pairs
                epochs_to_dataset_creation = args.epochs_per_dataset
            
            augmented_train_dataset.reshuffle(seed=epoch, need_props=False)
            epochs_to_dataset_creation -= 1
            train(
                model=model,
                train_dataset=augmented_train_dataset,
                criterion=criterion,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
                original_parameter_vector=original_parameter_vector,
                parameter_crit=parameter_crit,
                parameter_crit_weight=args.l2_diff_weight
            )
            val_loss = validate(
                model=model,
                val_dataset=val_dataset,
                criterion=criterion
            )
            os.makedirs(os.path.join(args.save_dir, 'epoch' + str(epoch)), exist_ok=True)
            augmented_train_dataset.save(os.path.join(args.save_dir, 'epoch' + str(epoch), 'train_pairs.csv'))
            save_model(
                model=model,
                i2s=train_dataset.i2s,
                path=os.path.join(args.save_dir, 'epoch' + str(epoch), 'val_loss_{}.pt'.format(val_loss))
            )
            predict(
                model=model,
                predict_dataset=predict_dataset,
                save_dir=os.path.join(args.save_dir, 'epoch' + str(epoch)),
                args=args,
                chemprop_predictor=chemprop_predictor if not args.no_predictor_at_val else None,
                sample=not args.greedy_prediction,
                num_predictions=args.val_num_predictions,
                print_filter_frac=args.print_filter_frac
            )
            evaluate(
                pred_smiles_dir=os.path.join(args.save_dir, 'epoch' + str(epoch)),
                train_path=args.train_path,
                val_path=args.val_path,
                checkpoint_dir=args.chemprop_dir,
                computed_prop=args.computed_prop,
                prop_min=args.prop_min,
                sim_thresholds=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                chemprop_predictor=chemprop_predictor,
                prop_max=args.prop_max,
                unconditional=args.unconditional
            )
            scheduler.step()
    
    # for convenient evaluation
    os.makedirs(os.path.join(args.save_dir, 'final_eval'), exist_ok=True)
    test_dataset = SourceDataset(
        path=args.test_path,
        i2s=train_dataset.i2s,
        s2i=train_dataset.s2i,
        pad_index=train_dataset.pad_index,
        start_index=train_dataset.start_index,
        end_index=train_dataset.end_index,
        batch_size=args.batch_size
    )
    predict(
        model=model,
        predict_dataset=test_dataset,
        save_dir=os.path.join(args.save_dir, 'final_eval'),
        args=args,
        chemprop_predictor=chemprop_predictor if not args.no_predictor_at_val else None,
        sample=not args.greedy_prediction,
        num_predictions=args.val_num_predictions,
        print_filter_frac=args.print_filter_frac
    )
    if args.final_eval_chemprop_dir is not None:
        args.computed_prop = None
        args.chemprop_dir = args.final_eval_chemprop_dir
        chemprop_predictor = ChempropPredictor(args)
    if args.final_eval_computed_prop is not None:
        args.chemprop_dir = None
        args.computed_prop = args.final_eval_computed_prop
        chemprop_predictor = ChempropPredictor(args)
    evaluate(
        pred_smiles_dir=os.path.join(args.save_dir, 'final_eval'),
        train_path=args.train_path,
        val_path=args.test_path,
        checkpoint_dir=args.chemprop_dir,
        computed_prop=args.computed_prop,
        prop_min=args.prop_min,
        sim_thresholds=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        chemprop_predictor=chemprop_predictor,
        prop_max=args.prop_max,
        unconditional=args.unconditional
    )
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/drd2/train_pair.txt')
    parser.add_argument('--val_path', type=str, default='../data/drd2/valid.txt')
    parser.add_argument('--test_path', type=str, default='../data/drd2/test.txt')
    parser.add_argument('--extra_precursors_path', type=str)
    parser.add_argument('--save_dir', type=str, default='../ckpt')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--checkpoint_dir', type=str, help='convenient alt way to specify ckpt; just walk the dir until you find a ckpt')

    parser.add_argument('--keep_translations', action='store_true', default=False, help='keep growing the self training dataset')
    parser.add_argument('--replace_old_dataset', action='store_true', default=False, help='do not keep using gold data')

    parser.add_argument('--unconditional', action='store_true', default=False, 
                        help='unconditional setting. you should feed the same datasets as for conditional; it just ignores input')
    parser.add_argument('--unconditional_vae', action='store_true', default=False, help='use vae in unconditional setting')
    parser.add_argument('--l2_diff_weight', type=float, default=0.0,
                        help='weight for for L2 difference penalty in weight vector between current model and pre-augmentation model')
    parser.add_argument('--dupe', action='store_true', default=False, help='duplicates allowed in unconditional self training')

    parser.add_argument('--val_num_predictions', type=int, default=20, help='Z in paper')
    parser.add_argument('--max_translations_per_pred', type=int, default=10, 
                        help='max num translations before giving up and picking one not above chemprop predictor cutoff. L in paper')
    parser.add_argument('--greedy_prediction', action='store_true', default=False)
    parser.add_argument('--no_conditional_vae', action='store_true', default=False)
    parser.add_argument('--no_predictor_at_val', action='store_true', default=False)
    parser.add_argument('--no_filter', action='store_true', default=False, help='ablation with no filter at all')
    parser.add_argument('--print_filter_frac', action='store_true', default=False, help='print fraction of samples passing filter at pred time')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--vae_latent_dim', type=int, default=30)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='Weight of the KL term of the loss')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='gradient clipping')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='Learning rate')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--self_train_epochs', type=int, default=0)
    parser.add_argument('--epochs_per_dataset', type=int, default=1, help='Epochs to wait between regenerating dataset')

    parser.add_argument('--num_translations_per_input', type=int, default=200, help='C in paper')
    parser.add_argument('--train_max_data', type=int, default=None)
    parser.add_argument('--self_train_max_data', type=int, default=None)
    parser.add_argument('--k', type=int, default=4, help='K in paper')
    parser.add_argument('--morgan_similarity_threshold', type=float, default=0.4)
    parser.add_argument('--chemprop_dir', type=str, default=None)
    parser.add_argument('--computed_prop', type=str, choices=['penalized_logp', 'logp', 'qed', 'sascore', 'drd2'], default=None,
                        help='Computed property to evaluate')
    parser.add_argument('--final_eval_chemprop_dir', type=str, default=None,
                        help='Chemprop dir to use at test time')
    parser.add_argument('--final_eval_computed_prop', type=str, choices=['penalized_logp', 'logp', 'qed', 'sascore', 'drd2'], default=None,
                        help='Computed property to evaluate at final test time')
    parser.add_argument('--prop_min', type=float, default=-float('inf'), help='min property of translation to be considered successful')
    parser.add_argument('--prop_max', type=float, default=None, help='max property of translation to be considered successful')
    parser.add_argument('--neg_threshold', action='store_true', default=False, help='flip sign of loaded property predictor')
    parser.add_argument('--prop_index', type=int, default=0, help='index of desired property from chemprop predictor')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--evaluate_every', type=int, default=1, help='evaluate every X epochs')

    args = parser.parse_args()

    if args.self_train_epochs > 0:
        num_props = 0
        for arg in [args.chemprop_dir, args.computed_prop]:
            if arg is not None:
                num_props += 1
        assert num_props == 1

    main(args)
