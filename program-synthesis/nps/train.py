# External imports
import json
import logging
import os
import random
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import cPickle as pickle
import datetime

from pathlib import Path
from tqdm import tqdm

# Project imports
from nps.data import load_input_file, get_minibatch, shuffle_dataset
from nps.network import IOs2Seq
from nps.reinforce import EnvironmentClasses, RewardCombinationFun
from nps.training_functions import (do_supervised_minibatch,
                                    do_syntax_weighted_minibatch,
                                    do_rl_minibatch,
                                    do_rl_minibatch_two_steps,
                                    do_beam_rl)
from nps.evaluate import evaluate_model, evaluate_model_parallel
from syntax.checker import PySyntaxChecker
from karel.consistency import Simulator

class TrainSignal(object):
    SUPERVISED = "supervised"
    RL = "rl"
    BEAM_RL = "beam_rl"

signals = ["supervised", "rl", "beam_rl"]

def add_train_cli_args(parser):
    train_group = parser.add_argument_group("Training",
                                            description="Training options")
    train_group.add_argument('--signal', type=str,
                             choices=signals,
                             default=signals[0],
                             help="Where to get gradients from"
                             "Default: %(default)s")
    train_group.add_argument('--nb_ios', type=int,
                             default=5)
    train_group.add_argument('--nb_epochs', type=int,
                             default=2,
                             help="How many epochs to train the model for. "
                             "Default: %(default)s")
    train_group.add_argument('--nb_st_epochs', type=int,
                             default=2,
                             help="How many self train epochs to train the model for. "
                             "Default: %(default)s")
    train_group.add_argument('--self_train_max_attempts', type=int, default=2, 
                        help='max tries to find k new targets')
    train_group.add_argument('--self_train_use_beam', action='store_true')
    train_group.add_argument('--self_train_only_new_samples', action='store_true')
    train_group.add_argument('--epochs_per_dataset', type=int, default=1, 
                        help='number of epochs to use a self train generated dataset before remaking')
    train_group.add_argument('--k', type=int, default=2)
    train_group.add_argument('--small_epochs_per_full', type=int, default=1, 
                             help='if you use Z, code uses 1/Z of data per epoch. will partition data over Z epochs')
    train_group.add_argument('--optim_alg', type=str,
                             default='Adam',
                             choices=['Adam', 'RMSprop', 'SGD'],
                             help="What optimization algorithm to use. "
                             "Default: %(default)s")
    train_group.add_argument('--batch_size', type=int,
                             default=32,
                             help="Batch Size for the optimization. "
                             "Default: %(default)s")
    train_group.add_argument('--learning_rate', type=float,
                             default=1e-3,
                             help="Learning rate for the optimization. "
                             "Default: %(default)s")
    train_group.add_argument("--train_file", type=str,
                             default="data/1m_6ex_karel/train.json",
                             help="Path to the training data. "
                             " Default: %(default)s")
    train_group.add_argument("--chunk_dir", type=str,
                             default=None,
                             help="Path to the training data. "
                             " Default: %(default)s")
    train_group.add_argument("--val_file", type=str,
                             default="data/1m_6ex_karel/val.json",
                             help="Path to the validation data. "
                             " Default: %(default)s")
    train_group.add_argument("--vocab", type=str,
                             default="data/1m_6ex_karel/new_vocab.vocab",
                             help="Path to the output vocabulary."
                             " Default: %(default)s")
    train_group.add_argument("--nb_samples", type=int,
                             default=0,
                             help="Max number of samples to look at."
                             "If 0, look at the whole dataset.")
    train_group.add_argument("--result_folder", type=str,
                             default="exps/fake_run",
                             help="Where to store the results. "
                             " Default: %(default)s")
    train_group.add_argument("--init_weights", type=str,
                             default=None)
    train_group.add_argument("--use_grammar", action="store_true")
    train_group.add_argument('--beta', type=float,
                             default=1e-3,
                             help="Gain applied to syntax loss. "
                             "Default: %(default)s")
    train_group.add_argument("--val_frequency", type=int,
                             default=1,
                             help="Frequency (in epochs) of validation.")

    rl_group = parser.add_argument_group("RL-specific training options")
    rl_group.add_argument("--environment", type=str,
                          choices=EnvironmentClasses.keys(),
                          default="BlackBoxGeneralization",
                          help="What type of environment to get a reward from"
                          "Default: %(default)s.")
    rl_group.add_argument("--reward_comb", type=str,
                          choices=RewardCombinationFun.keys(),
                          default="RenormExpected",
                          help="How to aggregate the reward over several samples.")
    rl_group.add_argument('--nb_rollouts', type=int,
                          default=100,
                          help="When using RL,"
                          "how many trajectories to sample per example."
                          "Default: %(default)s")
    rl_group.add_argument('--rl_beam', type=int,
                          default=50,
                          help="Size of the beam when doing reward"
                          " maximization over the beam."
                          "Default: %(default)s")
    rl_group.add_argument('--rl_inner_batch', type=int,
                          default=2,
                          help="Size of the batch on expanded candidates")
    rl_group.add_argument('--rl_use_ref', action="store_true")

def train_seq2seq_model(
        # Optimization
        signal, nb_ios, nb_epochs, nb_st_epochs, k, c, optim_alg,
        batch_size, learning_rate, use_grammar, beta, val_frequency,
        # Model
        kernel_size, conv_stack, fc_stack,
        tgt_embedding_size, lstm_hidden_size, nb_lstm_layers,
        learn_syntax,
        # RL specific options
        environment, reward_comb, nb_rollouts,
        rl_beam, rl_inner_batch, rl_use_ref,
        # What to train
        train_file, val_file, vocab_file, nb_samples, initialisation,
        # Where to write results
        result_folder, args_dict,
        # Run options
        use_cuda, log_frequency, self_train_use_beam, epochs_per_dataset, self_train_only_new_samples, chunk_dir, small_epochs_per_full):

    #############################
    # Admin / Bookkeeping stuff #
    #############################
    # Creating the results directory
    result_dir = Path(result_folder)
    if not result_dir.exists():
        os.makedirs(str(result_dir))
    else:
        # The result directory exists. Let's check whether or not all of our
        # work has already been done.

        # The sign of all the works being done would be the model after the
        # last epoch, let's check if it's here
        last_epoch_model_path = result_dir / "Weights" / ("weights_%d.model" % (nb_epochs - 1))
        if last_epoch_model_path.exists():
            print("{} already exists -- skipping this training".format(last_epoch_model_path))
            return


    # Dumping the arguments
    args_dump_path = result_dir / "args.json"
    with open(str(args_dump_path), "w") as args_dump_file:
        json.dump(args_dict, args_dump_file, indent=2)
    # Setting up the logs
    log_file = result_dir / "logs.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=str(log_file),
        filemode='w'
    )
    train_loss_path = result_dir / "train_loss.json"
    models_dir = result_dir / "Weights"
    if not models_dir.exists():
        os.makedirs(str(models_dir))
        time.sleep(1)  # Let some time for the dir to be created

    #####################################
    # Load Model / Dataset / Vocabulary #
    #####################################
    # Load-up the dataset
    dataset, vocab = load_input_file(train_file, vocab_file)

    if use_grammar:
        syntax_checker = PySyntaxChecker(vocab["tkn2idx"], use_cuda)
    # Reduce the number of samples in the dataset, if needed
    if nb_samples > 0:
        # Randomize the dataset to shuffle it, because I'm not sure that there
        # is no meaning in the ordering of the samples
        random.seed(0)
        dataset = shuffle_dataset(dataset, batch_size)
        dataset = {
            'sources' : dataset['sources'][:nb_samples],
            'targets' : dataset['targets'][:nb_samples],
        }

    vocabulary_size = len(vocab["tkn2idx"])
    if initialisation is None:
        # Create the model
        model = IOs2Seq(kernel_size, conv_stack, fc_stack,
                        vocabulary_size, tgt_embedding_size,
                        lstm_hidden_size, nb_lstm_layers,
                        learn_syntax)
        # Dump initial weights
        path_to_ini_weight_dump = models_dir / "ini_weights.model"
        with open(str(path_to_ini_weight_dump), "wb") as weight_file:
            torch.save(model, weight_file)
    else:
        model = torch.load(initialisation,
                           map_location=lambda storage, loc: storage)
    print('done creating/loading model')
    if use_grammar:
        model.set_syntax_checker(syntax_checker)
    tgt_start = vocab["tkn2idx"]["<s>"]
    tgt_end = vocab["tkn2idx"]["m)"]
    tgt_pad = vocab["tkn2idx"]["<pad>"]

    ############################################
    # Setup Loss / Optimizer / Eventual Critic #
    ############################################
    if signal == TrainSignal.SUPERVISED:
        # Create a mask to not penalize bad prediction on the padding
        weight_mask = torch.ones(vocabulary_size)
        weight_mask[tgt_pad] = 0
        # Setup the criterion
        loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
    elif signal == TrainSignal.RL or signal == TrainSignal.BEAM_RL:
        simulator = Simulator(vocab["idx2tkn"])

        if signal == TrainSignal.BEAM_RL:
            reward_comb_fun = RewardCombinationFun[reward_comb]
    else:
        raise Exception("Unknown TrainingSignal.")

    if use_cuda:
        model.cuda()
        if signal == TrainSignal.SUPERVISED:
            loss_criterion.cuda()

    # Setup the optimizers
    optimizer_cls = getattr(optim, optim_alg)
    optimizer = optimizer_cls(model.parameters(),
                              lr=learning_rate)

    #####################
    # ################# #
    # # Training Loop # #
    # ################# #
    #####################
    losses = []
    recent_losses = []
    best_val_acc = np.NINF
    # base_dataset = dataset
    path_to_weight_dump = initialisation
    for epoch_idx in range(0, nb_epochs + nb_st_epochs):

        # DATA AUGMENTATION
        if epoch_idx >= nb_epochs: # should start self training epochs
            print('beginning data augmentation epoch')
            print(str(datetime.datetime.now()))
            assert k > 0
            if (epoch_idx - nb_epochs) % epochs_per_dataset == 0: # resample the dataset every so many epochs
                print('remaking dataset')
                if chunk_dir is not None:
                    train_chunks = []
                    for _, _, filenames in os.walk(chunk_dir):
                        filenames = sorted(filenames)
                        for filename in filenames:
                            if '.json' in filename:
                                train_chunks.append(os.path.join(chunk_dir, filename))
                else:
                    train_chunks = [train_file]
                if (epoch_idx - nb_epochs) % (small_epochs_per_full * epochs_per_dataset) == 0:
                    all_chunk_idxs = list(range(len(train_chunks)))
                    random.shuffle(all_chunk_idxs)
                    chunk_partition = []
                    for i in range(small_epochs_per_full):
                        partition_start_idx = int(i*len(all_chunk_idxs) / small_epochs_per_full)
                        partition_end_idx = int((i+1)*len(all_chunk_idxs) / small_epochs_per_full)
                        chunk_partition.append(all_chunk_idxs[partition_start_idx:partition_end_idx])
                for chunk_idx, train_chunk_file in enumerate(train_chunks):
                    if chunk_idx not in chunk_partition[int(epoch_idx / epochs_per_dataset) % small_epochs_per_full]: # use 1/10 of chunks each epoch, as the dataset is large
                        continue
                    print('chunk' + str(chunk_idx))
                    dataset = None # free memory?
                    dataset, _ = load_input_file(train_chunk_file, vocab_file)
                    out_path = str(result_dir / ("eval/data_augment_epoch_" + str(epoch_idx) + "_chunk_" + str(chunk_idx) + "/_train_.txt"))
                    minibatch_cache_path = str(result_dir / ("eval/data_augment_minibatch/chunk_" + str(chunk_idx)))
                    results, _ = evaluate_model(str(path_to_weight_dump), vocab_file,
                                            train_chunk_file, 5, 0, use_grammar,
                                            out_path, 100, c, 5*batch_size,
                                            use_cuda, False, True, self_train_use_beam, k, minibatch_cache_path=minibatch_cache_path) # k: find k not including original target
                    # results is a len_train x c x (program_idxs, successful_generation_bool)
                    # add up to k most likely correct translations to the dataset
                    assert len(results) == len(dataset['sources'])
                    new_dataset = {'sources': [], 'targets': []}
                    for train_idx in range(len(dataset['sources'])):
                        source = dataset['sources'][train_idx]
                        base_target = dataset['targets'][train_idx]
                        new_targets = []
                        for program_idxs in results[train_idx]:
                            if program_idxs != base_target: # found a generalization that's not the original target
                                new_targets.append(program_idxs)
                                if len(new_targets) >= k:
                                    break
                        if len(new_targets) < k:
                            new_targets += [base_target for _ in range(k - len(new_targets))]
                        new_dataset['sources'] += [source for _ in range(k)]
                        new_dataset['targets'] += new_targets

                    # checking up to 1000 to ensure correctness during debugging
                    for i in range(min(1000, len(new_dataset['sources']))):
                        batch = get_minibatch(new_dataset, i, 1, tgt_start, tgt_end, tgt_pad, nb_ios, shuffle=True)
                        inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list, out_tgt_seq, inp_worlds, out_worlds, targets, inp_test_worlds, out_test_worlds = batch
                        simulator = Simulator(vocab['idx2tkn'])
                        _, cand_prog = simulator.get_prog_ast(targets[0])
                        res_emu = simulator.run_prog(cand_prog, inp_worlds[0][0])
                        assert (res_emu.status == 'OK') and (not res_emu.crashed) and (res_emu.outgrid == out_worlds[0][0])
                    
                    chunk_data_save_path = out_path = str(result_dir / ("eval/data_augment_epoch_" + str(epoch_idx) + "/_" + str(chunk_idx) + "_chunk_data_.txt"))
                    if not os.path.exists(os.path.dirname(chunk_data_save_path)):
                        os.makedirs(os.path.dirname(chunk_data_save_path))
                    with open(chunk_data_save_path, 'wb') as f:
                        pickle.dump(new_dataset, f)

                print('reloading and combining datasets')
                dataset = {'sources': [], 'targets': []}
                if not self_train_only_new_samples:
                    for chunk_idx, train_chunk_file in enumerate(train_chunks):
                        if chunk_idx not in chunk_partition[int(epoch_idx / epochs_per_dataset) % small_epochs_per_full]: # use 1/10 of chunks each epoch, as the dataset is large
                            continue
                        chunk_dataset, _ = load_input_file(train_chunk_file, vocab_file)
                        dataset['sources'] += chunk_dataset['sources']
                        dataset['targets'] += chunk_dataset['targets']
                for chunk_idx in range(len(train_chunks)):
                    if chunk_idx not in chunk_partition[int(epoch_idx / epochs_per_dataset) % small_epochs_per_full]: # use 1/10 of chunks each epoch, as the dataset is large
                        continue
                    chunk_data_save_path = out_path = str(result_dir / ("eval/data_augment_epoch_" + str(epoch_idx) + "/_" + str(chunk_idx) + "_chunk_data_.txt"))
                    with open(chunk_data_save_path, 'rb') as f:
                        chunk_new_dataset = pickle.load(f)
                    dataset['sources'] += chunk_new_dataset['sources']
                    dataset['targets'] += chunk_new_dataset['targets']

        nb_ios_for_epoch = nb_ios
        # This is definitely not the most efficient way to do it but oh well
        dataset = shuffle_dataset(dataset, batch_size)
        print('beginning training')
        for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):

            batch_idx = int(sp_idx/batch_size)
            optimizer.zero_grad()

            if signal == TrainSignal.SUPERVISED:
                inp_grids, out_grids, \
                    in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
                    _, _, _, _, _ = get_minibatch(dataset, sp_idx, batch_size,
                                                  tgt_start, tgt_end, tgt_pad,
                                                  nb_ios_for_epoch)
                if use_cuda:
                    inp_grids, out_grids = inp_grids.cuda(), out_grids.cuda()
                    in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
                if learn_syntax:
                    minibatch_loss = do_syntax_weighted_minibatch(model,
                                                                  inp_grids, out_grids,
                                                                  in_tgt_seq, in_tgt_seq_list,
                                                                  out_tgt_seq,
                                                                  loss_criterion, beta)
                else:
                    minibatch_loss = do_supervised_minibatch(model,
                                                             inp_grids, out_grids,
                                                             in_tgt_seq, in_tgt_seq_list,
                                                             out_tgt_seq, loss_criterion)
                recent_losses.append(minibatch_loss)
            elif signal == TrainSignal.RL or signal == TrainSignal.BEAM_RL:
                inp_grids, out_grids, \
                    _, _, _, \
                    inp_worlds, out_worlds, \
                    targets, \
                    inp_test_worlds, out_test_worlds = get_minibatch(dataset, sp_idx, batch_size,
                                                                     tgt_start, tgt_end, tgt_pad,
                                                                     nb_ios_for_epoch)
                if use_cuda:
                    inp_grids, out_grids = inp_grids.cuda(), out_grids.cuda()
                # We use 1/nb_rollouts as the reward to normalize wrt the
                # size of the rollouts
                if signal == TrainSignal.RL:
                    reward_norm = 1 / float(nb_rollouts)
                elif signal == TrainSignal.BEAM_RL:
                    reward_norm = 1
                else:
                    raise NotImplementedError("Unknown training signal")

                lens = [len(target) for target in targets]
                max_len = max(lens) + 10
                env_cls = EnvironmentClasses[environment]
                if "Consistency" in environment:
                    envs = [env_cls(reward_norm, trg_prog, sp_inp_worlds, sp_out_worlds, simulator)
                            for trg_prog, sp_inp_worlds, sp_out_worlds
                            in zip(targets, inp_worlds, out_worlds)]
                elif "Generalization" or "Perf" in environment:
                    envs = [env_cls(reward_norm, trg_prog, sp_inp_test_worlds, sp_out_test_worlds, simulator )
                            for trg_prog, sp_inp_test_worlds, sp_out_test_worlds
                            in zip(targets, inp_test_worlds, out_test_worlds)]
                else:
                    raise NotImplementedError("Unknown environment type")

                if signal == TrainSignal.RL:
                    minibatch_reward = do_rl_minibatch(model,
                                                       inp_grids, out_grids,
                                                       envs,
                                                       tgt_start, tgt_end, max_len,
                                                       nb_rollouts)
                    # minibatch_reward = do_rl_minibatch_two_steps(model,
                    #                                              inp_grids, out_grids,
                    #                                              envs,
                    #                                              tgt_start, tgt_end, tgt_pad,
                    #                                              max_len, nb_rollouts,
                    #                                              rl_inner_batch)
                elif signal == TrainSignal.BEAM_RL:
                    minibatch_reward = do_beam_rl(model,
                                                  inp_grids, out_grids, targets,
                                                  envs, reward_comb_fun,
                                                  tgt_start, tgt_end, tgt_pad,
                                                  max_len, rl_beam, rl_inner_batch, rl_use_ref)
                else:
                    raise NotImplementedError("Unknown Environment type")
                recent_losses.append(minibatch_reward)
            else:
                raise NotImplementedError("Unknown Training method")
            optimizer.step()
            if (batch_idx % log_frequency == log_frequency-1 and len(recent_losses) > 0) or \
               (len(dataset["sources"]) - sp_idx ) < batch_size:
                logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                    epoch_idx, batch_idx, sum(recent_losses)/len(recent_losses))
                )
                losses.extend(recent_losses)
                recent_losses = []
                # Dump the training losses
                with open(str(train_loss_path), "w") as train_loss_file:
                    json.dump(losses, train_loss_file, indent=2)

                if signal == TrainSignal.BEAM_RL:
                    # RL is much slower so we dump more frequently
                    path_to_weight_dump = models_dir / ("weights_%d.model" % epoch_idx)
                    with open(str(path_to_weight_dump), "wb") as weight_file:
                        # Needs to be in cpu mode to dump, otherwise will be annoying to load
                        if use_cuda:
                            model.cpu()
                        torch.save(model, weight_file)
                        if use_cuda:
                            model.cuda()

        # Dump the weights at the end of the epoch
        path_to_weight_dump = models_dir / ("weights_%d.model" % epoch_idx)
        with open(str(path_to_weight_dump), "wb") as weight_file:
            # Needs to be in cpu mode to dump, otherwise will be annoying to load
            if use_cuda:
                model.cpu()
            torch.save(model, weight_file)
            if use_cuda:
                model.cuda()
        previous_weight_dump = models_dir / ("weights_%d.model" % (epoch_idx-1))
        if previous_weight_dump.exists():
            if epoch_idx < nb_epochs: # don't delete the self training ones
                os.remove(str(previous_weight_dump))
        # Dump the training losses
        with open(str(train_loss_path), "w") as train_loss_file:
            json.dump(losses, train_loss_file, indent=2)

        logging.info("Done with epoch %d." % epoch_idx)

        if (epoch_idx+1) % val_frequency == 0 or (epoch_idx+1) == nb_epochs:
            # Evaluate the model on the validation set
            out_path = str(result_dir / ("eval/epoch_%d/val_.txt" % epoch_idx))
            val_acc = evaluate_model(str(path_to_weight_dump), vocab_file,
                                     val_file, 5, 0, use_grammar,
                                     out_path, 100, 50, batch_size,
                                     use_cuda, False)
            logging.info("Epoch : %d ValidationAccuracy : %f." % (epoch_idx, val_acc))
            if val_acc > best_val_acc:
                logging.info("Epoch : %d ValidationBest : %f." % (epoch_idx, val_acc))
                best_val_acc = val_acc
                path_to_weight_dump = models_dir / "best.model"
                with open(str(path_to_weight_dump), "wb") as weight_file:
                    # Needs to be in cpu mode to dump, otherwise will be annoying to load
                    if use_cuda:
                        model.cpu()
                    torch.save(model, weight_file)
                    if use_cuda:
                        model.cuda()
