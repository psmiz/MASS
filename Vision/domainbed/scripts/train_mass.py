# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
from tqdm import tqdm

sys.path.append("/mnt/lustre/bli/projects/EIL/domainbed")

# import wandb
import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision

from domainbed import algorithms_mass as algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--drop_out', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="../output/domainbed")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    
    # MASS (Conditional Parameter Distribution) arguments
    parser.add_argument('--enable_mass', action='store_true', help='Enable MASS for dynamic expert expansion')
    parser.add_argument('--rm_threshold', type=float, default=0.70, help='Redundancy measurement threshold for MASS')
    parser.add_argument('--mass_warmup_steps', type=int, default=100, help='Warmup steps before starting MASS expansion')
    parser.add_argument('--mass_window_size', type=int, default=200, help='Window size for MASS statistics collection')
    parser.add_argument('--mass_p_threshold', type=float, default=0.01, help='P-value threshold for MASS expansion decision')
    parser.add_argument('--mass_similarity_threshold', type=float, default=0.001, help='Similarity threshold for MASS')
    parser.add_argument('--mass_expansion_patience', type=int, default=3, help='Patience for MASS expansion before stopping')
    parser.add_argument('--mass_redundancy_weight', type=float, default=0.01, help='Weight for MASS redundancy regularization loss')
    
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    args.output_dir = os.path.join(args.output_dir, args.dataset, str(args.test_envs), args.algorithm, args.exp_name+str(args.hparams)+'_'+str(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))
    # if "Debug" not in args.dataset:
    #     wandb.init(project="sparse-moe",
    #                entity='drluodian',
    #                config={'dataset': args.dataset,
    #                        'algorithm': args.algorithm,
    #                        'test_envs': args.test_envs},
    #                settings=wandb.Settings(start_method="fork"))
    # wandb.init(settings=wandb.Settings(start_method='thread'))
    # print("Environment:")
    # print("\tPython: {}".format(sys.version.split(" ")[0]))
    # print("\tPyTorch: {}".format(torch.__version__))
    # print("\tTorchvision: {}".format(torchvision.__version__))
    # print("\tCUDA: {}".format(torch.version.cuda))
    # print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))
    #
    # print('Args:')
    # for k, v in sorted(vars(args).items()):
    #     print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size
    if args.drop_out is not None:
        hparams['drop_out'] = args.drop_out
    if args.lr is not None:
        hparams['lr'] = args.lr
    if args.weight_decay is not None:
        hparams['weight_decay'] = args.weight_decay

    # Transfer MASS arguments to hparams
    hparams['enable_mass'] = args.enable_mass
    hparams['rm_threshold'] = args.rm_threshold
    hparams['mass_warmup_steps'] = args.mass_warmup_steps
    hparams['mass_window_size'] = args.mass_window_size
    hparams['mass_p_threshold'] = args.mass_p_threshold
    hparams['mass_similarity_threshold'] = args.mass_similarity_threshold
    hparams['mass_expansion_patience'] = args.mass_expansion_patience
    hparams['mass_redundancy_weight'] = args.mass_redundancy_weight

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_, int(len(in_) * args.uda_holdout_fraction), misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    expansion_stop_step = int(n_steps * 0.1)
    global_stop_expansion = False
    
    # Initialize MASS parameters for vision models if using GMOE algorithm
    if args.algorithm == "GMOE" and hasattr(algorithm, 'model'):
        if hasattr(algorithm.model, 'blocks'):
            for i, block in enumerate(algorithm.model.blocks):
                if hasattr(block, 'mlp') and hasattr(block.mlp, '_init_mass_tracking'):
                    block.mlp.t_warmup = max(100, int(n_steps * 0.05))
                    block.mlp.window_size = max(200, int(n_steps * 0.09))
                    block.mlp._init_mass_tracking()
                    block.mlp._register_gradient_hooks()
                    print(f"MASS parameters reset for vision block {i} with t_warmup {block.mlp.t_warmup} and window_size {block.mlp.window_size}")
        elif hasattr(algorithm.model, 'vit') and hasattr(algorithm.model.vit.encoder, 'layer'):
            if hasattr(algorithm, 'args') and hasattr(algorithm.args, 'moe_layers'):
                for i in algorithm.args.moe_layers:
                    layer = algorithm.model.vit.encoder.layer[i]
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_init_mass_tracking'):
                        layer.mlp.t_warmup = max(100, int(n_steps * 0.05))
                        layer.mlp.window_size = max(200, int(n_steps * 0.09))
                        layer.mlp._init_mass_tracking()
                        layer.mlp._register_gradient_hooks()
                        print(f"MASS parameters reset for vision layer {i} with t_warmup {layer.mlp.t_warmup} and window_size {layer.mlp.window_size}")

    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device) for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        
        expansion_phase = args.algorithm == "GMOE" and hasattr(algorithm, 'model') and not global_stop_expansion
        step_vals = algorithm.update(minibatches_device, expansion_phase=expansion_phase, current_step=step)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        
        # Check for stopping expansion (10% of total steps)
        if expansion_phase:
            global_stop_expansion = step + 1 >= expansion_stop_step
            if global_stop_expansion:
                print(f"Vision MASS: Expansion phase completed at step {step + 1}. Removing MASS hooks...")
                if hasattr(algorithm.model, 'blocks'):
                    for i, block in enumerate(algorithm.model.blocks):
                        if hasattr(block, 'mlp') and hasattr(block.mlp, 'disable_mass_tracking'):
                            block.mlp.disable_mass_tracking()
                elif hasattr(algorithm.model, 'vit') and hasattr(algorithm.model.vit.encoder, 'layer'):
                    if hasattr(algorithm, 'args') and hasattr(algorithm.args, 'moe_layers'):
                        for i in algorithm.args.moe_layers:
                            layer = algorithm.model.vit.encoder.layer[i]
                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'disable_mass_tracking'):
                                layer.mlp.disable_mass_tracking()

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name + '_acc'] = acc

            results['algorithm'] = args.algorithm
            results['dataset'] = args.dataset
            results['test_envs'] = args.test_envs
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            # if wandb.run:
            #     wandb.log(results)
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    # save_checkpoint('model.pkl')

    # Save MASS results and statistics
    if args.algorithm == "GMOE" and hasattr(algorithm, 'model'):
        mass_trace = {}
        final_avg_k = None
        final_K = None
        
        if hasattr(algorithm.model, 'blocks'):
            for i, block in enumerate(algorithm.model.blocks):
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'get_mass_info'):
                    mass_info = block.mlp.get_mass_info()
                    mass_trace[f'block_{i}'] = mass_info['mass_trace']
                    
                    # Get final statistics from the first MoE block
                    if final_K is None and hasattr(block.mlp, 'gates') and len(block.mlp.gates) > 0:
                        gate = block.mlp.gates[0]
                        if hasattr(gate, 'avg_k'):
                            final_avg_k = float(gate.avg_k)
                        final_K = int(block.mlp.num_global_experts)
                        break
        elif hasattr(algorithm.model, 'vit') and hasattr(algorithm.model.vit.encoder, 'layer'):
            if hasattr(algorithm, 'args') and hasattr(algorithm.args, 'moe_layers'):
                for i in algorithm.args.moe_layers:
                    layer = algorithm.model.vit.encoder.layer[i]
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_mass_info'):
                        mass_info = layer.mlp.get_mass_info()
                        mass_trace[f'layer_{i}'] = mass_info['mass_trace']
                        
                        # Get final statistics from the first MoE layer
                        if final_K is None and hasattr(layer.mlp, 'gates') and len(layer.mlp.gates) > 0:
                            gate = layer.mlp.gates[0]
                            if hasattr(gate, 'avg_k'):
                                final_avg_k = float(gate.avg_k)
                            final_K = int(layer.mlp.num_global_experts)
                            break
        
        # Save MASS trace
        if mass_trace:
            import pickle
            mass_trace_path = os.path.join(args.output_dir, "mass_trace.pkl")
            with open(mass_trace_path, "wb") as f:
                pickle.dump(mass_trace, f)
            print(f"Saved MASS trace to {mass_trace_path}")
            
        # Save final statistics
        if final_avg_k is not None and final_K is not None:
            avg_k_filename = os.path.join(args.output_dir, f"Final_K_{final_K}_Avg_k_{final_avg_k:.2f}.txt")
            with open(avg_k_filename, "w") as f:
                f.write(f'final_K: {final_K}\n')
                f.write(f'avg_k: {final_avg_k:.2f}\n')
            print(f"Final Vision MASS Statistics - K: {final_K}, Avg k: {final_avg_k:.2f}")

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
