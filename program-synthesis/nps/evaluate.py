from __future__ import division
# External imports
import json
import os
import torch
import cPickle as pickle
import datetime

from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

from nps.data import load_input_file, get_minibatch, save_minibatch_parallel, shuffle_dataset, make_minibatch_path, load_minibatch_parallel, make_decoded_path, save_decoded_parallel
from karel.consistency import Simulator
import pyximport
pyximport.install()
from syntax.checker import PySyntaxChecker

PARALLEL_LOAD_MINIBATCHES = 30

def add_eval_args(parser):
    parser.add_argument('--eval_nb_ios', type=int,
                        default=5)
    parser.add_argument('--use_grammar', action="store_true")
    parser.add_argument("--val_nb_samples", type=int,
                        default=0,
                        help="How many samples to use to compute the accuracy."
                        "Default: %(default)s, for all the dataset")

def add_beam_size_arg(parser):
    parser.add_argument("--eval_batch_size", type=int,
                        default=8)
    parser.add_argument("--beam_size", type=int,
                        default=10,
                        help="Size of the beam search. Default %(default)s")
    parser.add_argument("--top_k", type=int,
                        default=5,
                        help="How many candidates to return. Default %(default)s")

# evaluation loop for external filter
def find_generalizations(minibatch_cache_path, sp_idx, output_path, vocab_pickle, tgt_pad, num_successful_targets_per_precursor, top_k):
    with open(make_minibatch_path(minibatch_cache_path, sp_idx), 'rb') as f:
        inp_grids, out_grids, \
        in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
        inp_worlds, out_worlds, \
        _, \
        inp_test_worlds, out_test_worlds = pickle.load(f)

    with open(make_decoded_path(output_path, sp_idx), 'rb') as f:
        decoded = pickle.load(f)
    vocab = pickle.loads(vocab_pickle)
    simulator = Simulator(vocab["idx2tkn"])
    batch_results = []
    num_found_full = 0
    for batch_idx, (target, sp_decoded,
                    sp_input_worlds, sp_output_worlds,
                    sp_test_input_worlds, sp_test_output_worlds) in \
        enumerate(zip(out_tgt_seq.chunk(out_tgt_seq.size(0)), decoded,
                    inp_worlds, out_worlds,
                    inp_test_worlds, out_test_worlds)):

        num_success = 0
        current_results = []
        target = target.data.squeeze().numpy().tolist()
        target = [tkn_idx for tkn_idx in target if tkn_idx != tgt_pad]
        # Generalization
        found_full = False
        for rank, dec in enumerate(sp_decoded):
            pred = dec[-1]
            if pred == target:
                continue
            if pred in current_results:
                continue
            parse_success, cand_prog = simulator.get_prog_ast(pred) 
            if (not parse_success):
                continue
            generalizes = True
            for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                res_emu = simulator.run_prog(cand_prog, input_world)
                if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                    # This prediction is semantically incorrect.
                    generalizes = False
                    break
            for (input_world, output_world) in zip(sp_test_input_worlds, sp_test_output_worlds): # these are "test" worlds but really they're all used during training, it just samples 5 each time
                res_emu = simulator.run_prog(cand_prog, input_world)
                if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                    # This prediction is semantically incorrect.
                    generalizes = False
                    break
            if generalizes:
                current_results.append(pred)
                num_success += 1
                if num_success >= num_successful_targets_per_precursor:
                    found_full = True
                    break
        batch_results.append(current_results)
        if found_full:
            num_found_full += 1
    if num_found_full == 0:
        print('warning: num found full ' + str(num_found_full))
    return batch_results, (num_found_full, len(decoded))

def find_generalizations_parallel(args):
    return find_generalizations(*args)

def evaluate_model(model_weights,
                   vocabulary_path,
                   dataset_path,
                   nb_ios,
                   nb_samples,
                   use_grammar,
                   output_path,
                   beam_size,
                   top_k,
                   batch_size,
                   use_cuda,
                   dump_programs,
                   return_individual_results=False,
                   use_beam=True,
                   num_successful_targets_per_precursor=None,
                   preloaded_data=False,
                   minibatch_cache_path=None):
    if return_individual_results:
        print('beginning evaluate_model for data augmentation')
    all_outputs_path = []
    all_semantic_output_path = []
    all_syntax_output_path = []
    all_generalize_output_path = []
    pred_filter_path = 'pred_filter_top1_l%d.txt' % top_k
    pred_filter_path = output_path + pred_filter_path
    res_dir = os.path.dirname(output_path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for k in range(top_k):

        new_term = "exactmatch_top%d.txt" % (k+1)
        new_semantic_term = "semantic_top%d.txt" % (k+1)
        new_syntax_term = "syntax_top%d.txt" % (k+1)
        new_generalize_term = "fullgeneralize_top%d.txt" % (k+1)

        new_file_name = output_path + new_term
        new_semantic_file_name = output_path + new_semantic_term
        new_syntax_file_name = output_path + new_syntax_term
        new_generalize_file_name = output_path + new_generalize_term

        all_outputs_path.append(new_file_name)
        all_semantic_output_path.append(new_semantic_file_name)
        all_syntax_output_path.append(new_syntax_file_name)
        all_generalize_output_path.append(new_generalize_file_name)
    program_dump_path = os.path.join(res_dir, "generated")

    # if os.path.exists(all_outputs_path[0]):
    #     with open(all_outputs_path[0], "r") as out_file:
    #         out_file_content = out_file.read()
    #         print("Using cached result from {}".format(all_outputs_path[0]))
    #         print("Greedy select accuracy: {}".format(out_file_content))
    #         return

    # Load the vocabulary of the trained model
    if preloaded_data:
        dataset, vocab = pickle.loads(dataset_path), pickle.loads(vocabulary_path) # pre-loaded
    else:
        dataset, vocab = load_input_file(dataset_path, vocabulary_path)

    tgt_start = vocab["tkn2idx"]["<s>"]
    tgt_end = vocab["tkn2idx"]["m)"]
    tgt_pad = vocab["tkn2idx"]["<pad>"]

    simulator = Simulator(vocab["idx2tkn"])
    # Load the model
    if not use_cuda:
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/8
        # Is it failing?
        model = torch.load(model_weights, map_location=lambda storage, loc: storage)
    else:
        model = torch.load(model_weights)
        model.cuda()
    # And put it into evaluation mode
    model.eval()

    syntax_checker = PySyntaxChecker(vocab["tkn2idx"], use_cuda)
    if use_grammar:
        model.set_syntax_checker(syntax_checker)

    if beam_size == 1:
        top_k = 1
    nb_correct = [0 for _ in range(top_k)]
    nb_semantic_correct = [0 for _ in range(top_k)]
    nb_syntax_correct = [0 for _ in range(top_k)]
    nb_generalize_correct = [0 for _ in range(top_k)]
    nb_pred_filter_correct = 0
    total_nb = 0

    with torch.no_grad():
        if minibatch_cache_path is None or not os.path.isdir(minibatch_cache_path):
            if minibatch_cache_path is None:
                minibatch_cache_path = output_path
            dataset, sort_idx = shuffle_dataset(dataset, batch_size, randomize=False, return_sort_idx=True)
            if not os.path.isdir(minibatch_cache_path):
                os.makedirs(minibatch_cache_path)
            with open(os.path.join(minibatch_cache_path, 'dataset_sortidx.pkl'), 'wb') as f:
                pickle.dump((dataset, sort_idx), f)
            if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
            print('create inputs for saving minibatches')
            save_minibatch_inputs = []
            for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
                sources = dataset['sources'][sp_idx:sp_idx+batch_size]
                targets = dataset['targets'][sp_idx:sp_idx+batch_size]
                minibatch_output_path = make_minibatch_path(minibatch_cache_path, sp_idx)
                save_minibatch_inputs.append((sources, targets, minibatch_output_path, batch_size,
                                                                tgt_start, tgt_end, tgt_pad,
                                                                nb_ios, False, True))
            print('save minibatches')
            print(str(datetime.datetime.now()))
            for i in range(0, len(save_minibatch_inputs), 60): # to reset ram periodically...
                pool = Pool(processes=30, maxtasksperchild=1)
                pool.map(save_minibatch_parallel, save_minibatch_inputs[i:i+60], chunksize=1)
                pool.close()
            print(str(datetime.datetime.now()))
        else:
            with open(os.path.join(minibatch_cache_path, 'dataset_sortidx.pkl'), 'rb') as f:
                dataset, sort_idx = pickle.load(f)
        
        print('sample decodings from model')
        all_decoded = []
        load_count = 0
        pool = Pool(processes=30, maxtasksperchild=1)
        for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
            if load_count % PARALLEL_LOAD_MINIBATCHES == 0:
                minibatch_load_paths = []
                for i in range(PARALLEL_LOAD_MINIBATCHES):
                    mb_sp_idx = sp_idx + i * batch_size
                    if mb_sp_idx < len(dataset["sources"]):
                        minibatch_load_paths.append(make_minibatch_path(minibatch_cache_path, mb_sp_idx))
                loaded_minibatches = None
                loaded_minibatches = pool.map(load_minibatch_parallel, minibatch_load_paths, chunksize=1)
            minibatch = loaded_minibatches[load_count % PARALLEL_LOAD_MINIBATCHES]
            load_count += 1

            inp_grids, out_grids, \
            in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
            inp_worlds, out_worlds, \
            _, \
            inp_test_worlds, out_test_worlds = minibatch

            if return_individual_results:
                with open(make_minibatch_path(minibatch_cache_path, sp_idx), 'wb') as f:
                    pickle.dump(minibatch, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            max_len = out_tgt_seq.size(1) + 10
            if use_cuda:
                inp_grids, out_grids = inp_grids.cuda(), out_grids.cuda()
                in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()

            
            if dump_programs:
                decoder_logit, syntax_logit = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
                if syntax_logit is not None and model.decoder.learned_syntax_checker is not None:
                    syntax_logit = syntax_logit.cpu().data.numpy()
                    for n in range(in_tgt_seq.size(0)):
                        decoded_dump_dir = os.path.join(program_dump_path, str(n + sp_idx))
                        if not os.path.exists(decoded_dump_dir):
                            os.makedirs(decoded_dump_dir)
                        seq = in_tgt_seq.cpu().data.numpy()[n].tolist()
                        seq_len = seq.index(0) if 0 in seq else len(seq)
                        file_name = str(n) + "_learned_syntax"
                        norm_logit = syntax_logit[n,:seq_len]
                        norm_logit = np.log(-norm_logit)
                        norm_logit = 1 / (1 + np.exp(-norm_logit))
                        np.save(os.path.join(decoded_dump_dir, file_name), norm_logit)
                        ini_state = syntax_checker.get_initial_checker_state()
                        file_name = str(n) + "_manual_syntax"
                        mask = syntax_checker.get_sequence_mask(ini_state, seq).squeeze().cpu().numpy()[:seq_len]
                        np.save(os.path.join(decoded_dump_dir, file_name), mask)
                        file_name = str(n) + "_diff"
                        diff = mask.astype(float) - norm_logit
                        diff = (diff + 1) / 2 # remap to [0,1]
                        np.save(os.path.join(decoded_dump_dir, file_name), diff)

            decoded = model.beam_sample(inp_grids, out_grids,
                                        tgt_start, tgt_end, max_len,
                                        beam_size, top_k, True, use_beam)
            all_decoded.append((sp_idx, decoded))

            if load_count % PARALLEL_LOAD_MINIBATCHES == PARALLEL_LOAD_MINIBATCHES - 1:
                # save decodings
                decoded_load_paths = []
                for d_sp_idx, decoded in all_decoded:
                    decoded_load_paths.append(make_decoded_path(output_path, d_sp_idx))
                pool.map(save_decoded_parallel, [(decoded, path) for decoded, path in zip([tup[1] for tup in all_decoded], decoded_load_paths)], chunksize=1)
                all_decoded = []
        if len(all_decoded) > 0:
            # save decodings
            decoded_load_paths = []
            for d_sp_idx, decoded in all_decoded:
                decoded_load_paths.append(make_decoded_path(output_path, d_sp_idx))
            pool.map(save_decoded_parallel, [(decoded, path) for decoded, path in zip([tup[1] for tup in all_decoded], decoded_load_paths)], chunksize=1)
            all_decoded = []
        pool.close()

        if return_individual_results:
            print('done sampling from model, starting filter')
            generalization_inputs = []
            vocab_pickle = pickle.dumps(vocab, protocol=pickle.HIGHEST_PROTOCOL)
            for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
                generalization_inputs.append((minibatch_cache_path, sp_idx, output_path, vocab_pickle, tgt_pad, num_successful_targets_per_precursor, top_k))
            print('starting pool.map')
            pool = Pool(processes=30, maxtasksperchild=1)
            print(str(datetime.datetime.now()))
            all_batch_results = pool.map(find_generalizations_parallel, generalization_inputs, chunksize=1)
            print(str(datetime.datetime.now()))
            pool.close()
            full_results = []
            num_found_full = 0
            for batch_results, stats in all_batch_results:
                full_results += batch_results
                num_found_full += stats[0]
        else:
            for sp_idx in tqdm(range(0, len(dataset["sources"]), batch_size)):
                inp_grids, out_grids, \
                in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
                inp_worlds, out_worlds, \
                _, \
                inp_test_worlds, out_test_worlds = get_minibatch(dataset, sp_idx, batch_size,
                                                                tgt_start, tgt_end, tgt_pad,
                                                                nb_ios, shuffle=False, volatile_vars=True)

                max_len = out_tgt_seq.size(1) + 10
                # decoded = model.beam_sample(inp_grids, out_grids,
                #                             tgt_start, tgt_end, max_len,
                #                             beam_size, top_k)
                # decoded = all_decoded[int(sp_idx / batch_size)]
                with open(make_decoded_path(output_path, sp_idx), 'rb') as f:
                        decoded = pickle.load(f)
                for batch_idx, (target, sp_decoded,
                                sp_input_worlds, sp_output_worlds,
                                sp_test_input_worlds, sp_test_output_worlds) in \
                    enumerate(zip(out_tgt_seq.chunk(out_tgt_seq.size(0)), decoded,
                                inp_worlds, out_worlds,
                                inp_test_worlds, out_test_worlds)):

                    if return_individual_results:
                        current_results = []
                    total_nb += 1
                    target = target.data.squeeze().numpy().tolist()
                    target = [tkn_idx for tkn_idx in target if tkn_idx != tgt_pad]

                    if dump_programs:
                        decoded_dump_dir = os.path.join(program_dump_path, str(batch_idx + sp_idx))
                        if not os.path.exists(decoded_dump_dir):
                            os.makedirs(decoded_dump_dir)
                        write_program(os.path.join(decoded_dump_dir, "target"), target, vocab["idx2tkn"])
                        for rank, dec in enumerate(sp_decoded):
                            pred = dec[1]
                            ll = dec[0]
                            file_name = str(rank)+ " - " + str(ll)
                            write_program(os.path.join(decoded_dump_dir, file_name), pred, vocab["idx2tkn"])


                    # Exact matches
                    for rank, dec in enumerate(sp_decoded):
                        pred = dec[-1]
                        if pred == target:
                            # This prediction is correct. This means that we score for
                            # all the following scores
                            for top_idx in range(rank, top_k):
                                nb_correct[top_idx] += 1
                            break

                    # Semantic matches
                    for rank, dec in enumerate(sp_decoded):
                        pred = dec[-1]
                        parse_success, cand_prog = simulator.get_prog_ast(pred)
                        if (not parse_success):
                            continue
                        semantically_correct = True
                        for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                            res_emu = simulator.run_prog(cand_prog, input_world)
                            if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                                # This prediction is semantically incorrect.
                                semantically_correct = False
                                break
                        if semantically_correct:
                            # Score for all the following ranks
                            for top_idx in range(rank, top_k):
                                nb_semantic_correct[top_idx] += 1
                            break

                    # Generalization
                    found_full = False
                    pred_filter_success = 0
                    for rank, dec in enumerate(sp_decoded):
                        scored = False
                        num_success = 0
                        pred = dec[-1]
                        parse_success, cand_prog = simulator.get_prog_ast(pred)
                        if (not parse_success):
                            continue
                        generalizes = True
                        for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                            res_emu = simulator.run_prog(cand_prog, input_world)
                            if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                                # This prediction is semantically incorrect.
                                generalizes = False
                                break
                        if generalizes:
                            for (input_world, output_world) in zip(sp_test_input_worlds, sp_test_output_worlds):
                                res_emu = simulator.run_prog(cand_prog, input_world)
                                if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                                    # This prediction is semantically incorrect.
                                    generalizes = False
                                    pred_filter_success = -1 # we picked something that was syntactically correct and passed the inputs, but fails on test
                                    # NOTE: our setup is that we are allowed to filter as much as we want using the input test cases, but only get 1 try for the real "test" test cases.
                                    break
                        if return_individual_results and generalizes:
                            current_results.append(pred)
                            num_success += 1
                            if num_success >= num_successful_targets_per_precursor:
                                found_full = True
                                break
                        if generalizes and not scored:
                            if pred_filter_success != -1:
                                nb_pred_filter_correct += 1
                            # Score for all the following ranks
                            for top_idx in range(rank, top_k):
                                nb_generalize_correct[top_idx] += 1
                            if return_individual_results:
                                scored = True
                            else:
                                break
                    if return_individual_results:
                        full_results.append(current_results)
                        if found_full:
                            num_found_full += 1

                    # Correct syntaxes
                    for rank, dec in enumerate(sp_decoded):
                        pred = dec[-1]
                        parse_success, cand_prog = simulator.get_prog_ast(pred)
                        if parse_success:
                            for top_idx in range(rank, top_k):
                                nb_syntax_correct[top_idx] += 1
                            break

            with open(pred_filter_path, 'w') as pred_filter_file:
                pred_filter_file.write(str(100*nb_pred_filter_correct/total_nb))
            for k in range(top_k):
                with open(str(all_outputs_path[k]), "w") as res_file:
                    res_file.write(str(100*nb_correct[k]/total_nb))
                with open(str(all_semantic_output_path[k]), "w") as sem_res_file:
                    sem_res_file.write(str(100*nb_semantic_correct[k]/total_nb))
                with open(str(all_syntax_output_path[k]), "w") as stx_res_file:
                    stx_res_file.write(str(100*nb_syntax_correct[k]/total_nb))
                with open(str(all_generalize_output_path[k]), "w") as gen_res_file:
                    gen_res_file.write(str(100*nb_generalize_correct[k]/total_nb))

        if return_individual_results:
            print(str(datetime.datetime.now()))
            print('found full set of new targets for ' + str(num_found_full) + ' out of ' + str(len(full_results)))
            unsort_idx = [sort_idx.index(i) for i in range(len(sort_idx))]
            full_results = [full_results[u] for u in unsort_idx]
            return full_results, (num_found_full, len(full_results))
        else:
            semantic_at_one = 100*nb_semantic_correct[0]/total_nb
            return semantic_at_one

def evaluate_model_parallel(args):
    result = evaluate_model(*args)
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)

def write_program(path, tkn_idxs, vocab):
    program_tkns = [vocab[tkn_idx] for tkn_idx in tkn_idxs]

    indent = 0
    is_new_line = False
    with open(path, "w") as target_file:
        for tkn in program_tkns:
            if tkn in ["m(", "w(", "i(", "e(", "r("]:
                indent += 4
                target_file.write("\n"+" "*indent)
                target_file.write(tkn + " ")
                is_new_line = False
            elif tkn in ["m)", "w)", "i)", "e)", "r)"]:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                indent -= 4
                target_file.write(tkn)
                if indent < 0:
                    indent = 0
                is_new_line = True
            elif tkn in ["REPEAT"]:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                    is_new_line = False
                target_file.write(tkn + " ")
            else:
                if is_new_line:
                    target_file.write("\n"+" "*indent)
                    is_new_line = False
                target_file.write(tkn + " ")
        target_file.write("\n")
