# Improving Molecular Design by Stochastic Iterative Target Augmentation

Paper (ICML 2020): https://arxiv.org/abs/2002.04720

Implementation of VSeq2Seq and VSeq models for stochastic iterative target augmentation in molecular design.

Requirements: uses python 3.6, pytorch 0.4.1 (newer versions of python and pytorch should be ok), Chemprop (`https://github.com/chemprop`) for the proxy predictor, and RDKit for evaluation.

Data for QED and DRD2 datasets provided in `data/`, and example proxy predictors in `ckpt/proxy/`. 

# Example command for conditional setting (molecular optimization):

`python train.py --train_path data/qed/train_pairs.txt --val_path data/qed/valid.txt --test_path data/qed/test.txt --epochs 5 --self_train_epochs 10 --chemprop_dir ckpt/proxy/qed --final_eval_computed_prop qed --morgan_similarity_threshold 0.4 --prop_min 0.9 --save_dir ckpt/example_conditional_qed`

See `https://github.com/chemprop` for training your own proxy predictor. Alternatively, specify e.g. `--computed_prop qed` instead of `--chemprop_dir` to just use the ground truth (oracle) evaluator during iterative target augmentation. See the full model args with help strings at the bottom of `train.py`.

# Example command for unconditional setting (molecular generation):

`python train.py --train_path data/qed/train_pairs.txt --val_path data/qed/valid.txt --test_path data/qed/test.txt --epochs 1 --self_train_epochs 50 --chemprop_dir ckpt/proxy/qed --final_eval_computed_prop qed --morgan_similarity_threshold 0 --prop_min 0.9 --save_dir ckpt/example_unconditional_qed --unconditional --unconditional_vae --val_num_predictions 20000 --max_translations_per_pred 1 --replace_old_dataset`

(Similar interface as for conditional setting, but with some extra args at the end. Note that the validation and test sets are unused in this case.)