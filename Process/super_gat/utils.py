import itertools
from typing import Tuple, List, Any
import hashlib
import random
import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from termcolor import cprint
from ruamel.yaml import YAML
from itertools import islice
from torch import Tensor
import math

def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

def zeros(value: Any):
    constant(value, 0.)

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

def grouper(iterable, n):
    """https://stackoverflow.com/a/8991553"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def iter_window(seq, window_size=2, drop_last=False):
    """Returns a sliding window (of width n) over data from the iterable
        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        https://stackoverflow.com/a/6822773"""
    it = iter(seq)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
    if not drop_last:
        for i in range(1, len(result)):
            yield result[i:] + (None,) * i


def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def negative_sampling_numpy(edge_index_numpy: np.ndarray, num_nodes=None, num_neg_samples=None):
    num_neg_samples = num_neg_samples or edge_index_numpy.shape[1]

    # Handle '2*|edges| > num_nodes^2' case.
    num_neg_samples = min(num_neg_samples,
                          num_nodes * num_nodes - edge_index_numpy.shape[1])

    idx = (edge_index_numpy[0] * num_nodes + edge_index_numpy[1])

    rng = range(num_nodes ** 2)
    perm = np.asarray(random.sample(rng, num_neg_samples))
    mask = np.isin(perm, idx).astype(np.uint8)
    rest = mask.nonzero()[0]
    while np.prod(rest.shape) > 0:
        tmp = random.sample(rng, rest.shape[0])
        mask = np.isin(tmp, idx).astype(np.uint8)
        perm[rest] = tmp
        rest = rest[mask.nonzero()[0]]

    row, col = perm / num_nodes, perm % num_nodes
    return np.stack([row, col], axis=0)


def s_join(concat, lst):
    return concat.join([str(e) for e in lst])


def sigmoid(x):
    return float(np_sigmoid(x))


def np_sigmoid(x):
    return 1. / (1. + np.exp(-x))


def get_cartesian(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def to_one_hot(labels_integer_tensor: torch.Tensor, n_classes: int) -> np.ndarray:
    labels = labels_integer_tensor.cpu().numpy()
    return np.eye(n_classes)[labels]


def create_hash(o: dict):
    def preprocess(v):
        if isinstance(v, torch.Tensor):
            return v.shape
        else:
            return v

    sorted_keys = sorted(o.keys())
    strings = "/ ".join(["{}: {}".format(k, preprocess(o[k])) for k in sorted_keys])
    return hashlib.md5(strings.encode()).hexdigest()


def get_accuracy(preds_mat: np.ndarray, labels_mat: np.ndarray):
    return (np.sum(np.argmax(preds_mat, 1) == np.argmax(labels_mat, 1))
            / preds_mat.shape[0])


def get_roc_auc(preds_mat: np.ndarray, labels_mat: np.ndarray):
    return roc_auc_score(labels_mat, preds_mat)


def get_entropy_tensor(x: torch.Tensor, is_prob_dist=False):
    """
    :param x: tensor the shape of which is [*, C]
    :param is_prob_dist: if is_prob_dist == False: apply softmax
    :return: tensor the shape of which is [] (reduction=batchmean like F.kl_div)
    """
    x = x if is_prob_dist else F.softmax(x, dim=-1)
    prob_sum = x.size(0) if len(x.size()) > 1 else 1
    assert abs(x.sum() - prob_sum) < 1e-5, "{} is not {}".format(x.sum(), prob_sum)
    not_entropy_yet = x * torch.log(x)
    return -1.0 * not_entropy_yet.sum(dim=-1).mean()


def get_entropy_tensor_by_iter(x_list: List[torch.Tensor], is_prob_dist=False) -> torch.Tensor:
    """
    :param x_list: List of tensors [*, X, C_i]
    :param is_prob_dist:
    :return: [*]
    """
    entropy_list = []
    for x in x_list:
        try:
            entropy = get_entropy_tensor(x, is_prob_dist)  # []
            entropy_list.append(entropy)
        except AssertionError as e:
            print("Error: get_entropy_tensor_by_iter ({})".format(e))
    entropy_tensor = torch.stack(entropy_list)  # [*]
    return entropy_tensor.to(x_list[0].device)


def torch_log_stable(target_tensor, eps=1e-43):
    if not target_tensor.min() == 0:
        return torch.log(target_tensor)
    else:
        target_tensor[target_tensor == 0] = eps
        log_tensor = torch.log(target_tensor)
        return log_tensor


def get_kld_tensor_by_iter(pd_list: List[torch.Tensor], qd_list: List[torch.Tensor]) -> torch.Tensor:
    """
    :param pd_list: List of tensors [*, X, C_i]
    :param qd_list: List of tensors [*, X, C_i]
    :return: [*]
    """
    kld_list = []
    for pd, qd in zip(pd_list, qd_list):
        # input given is expected to contain log-probabilities.
        # The targets are given as probabilities (i.e. without taking the logarithm).
        log_pd = torch_log_stable(pd)
        # pd, qd: [X, C_i] -> []
        kld = F.kl_div(log_pd, qd, reduction="batchmean")  # []
        kld_list.append(kld)
    kld_tensor = torch.stack(kld_list)
    return kld_tensor.to(pd_list[0].device)


# GPU

def get_gpu_utility(gpu_id_or_ids: int or list) -> List[int]:
    if isinstance(gpu_id_or_ids, int):
        gpu_ids = [gpu_id_or_ids]
    else:
        gpu_ids = gpu_id_or_ids

    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")

    gpu_utilities = []
    for item in out_list:
        items = [x.strip() for x in item.split(':')]
        if len(items) == 2:
            key, val = items
            if key == "Minor Number":
                gpu_utilities.append(int(val))

    gpu_utilities = [g - min(gpu_utilities) for g in gpu_utilities]

    if len(gpu_utilities) < len(gpu_ids):
        raise EnvironmentError(
            "Cannot find all GPUs whose ids are {}, only found {} GPUs".format(gpu_ids, len(gpu_utilities)))
    else:
        return gpu_utilities


def get_free_gpu_names(num_gpus_total: int, threshold=30) -> List[str]:
    """
    :param num_gpus_total: total number of gpus
    :param threshold: Return GPUs the utilities of which is smaller than threshold.
    :return e.g. ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    """
    gpu_ids = list(range(num_gpus_total))
    gpu_utilities = get_gpu_utility(gpu_ids)
    return ["/device:GPU:{}".format(gid) for gid, utility in zip(gpu_ids, gpu_utilities) if utility <= threshold]


def get_free_gpu_names_safe(num_gpus_total: int, threshold=30, iteration=3) -> List[str]:
    gpu_set = set(get_free_gpu_names(num_gpus_total, threshold))
    for _ in range(iteration - 1):
        gpu_set = gpu_set.intersection(set(get_free_gpu_names(num_gpus_total, threshold)))
    return list(gpu_set)


def get_free_gpu_ids(num_gpus_total: int, threshold=30) -> List[int]:
    free_gpu_names = get_free_gpu_names(num_gpus_total=num_gpus_total, threshold=threshold)
    return [int(g.split(":")[-1]) for g in free_gpu_names]


def get_free_gpu_ids_safe(num_gpus_total: int, threshold=30) -> List[int]:
    free_gpu_names = get_free_gpu_names_safe(num_gpus_total=num_gpus_total, threshold=threshold)
    return [int(g.split(":")[-1]) for g in free_gpu_names]


def blind_other_gpus(num_gpus_total, num_gpus_to_use, is_safe=True, gpu_deny_list=None, **kwargs):
    if is_safe:
        free_gpu_ids = get_free_gpu_ids_safe(num_gpus_total, **kwargs)
    else:
        free_gpu_ids = get_free_gpu_ids(num_gpus_total, **kwargs)

    if gpu_deny_list is not None:
        free_gpu_ids = [g for g in free_gpu_ids if g not in gpu_deny_list]

    if free_gpu_ids:
        gpu_ids_to_use = random.sample(free_gpu_ids, num_gpus_to_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(n) for n in gpu_ids_to_use)
    else:
        gpu_ids_to_use = []

    return gpu_ids_to_use


# Others

def debug_with_exit(func):
    def wrapped(*args, **kwargs):
        print()
        cprint("===== DEBUG ON {}=====".format(func.__name__), "red", "on_yellow")
        func(*args, **kwargs)
        cprint("=====   END  =====", "red", "on_yellow")
        exit()

    return wrapped


def cprint_multi_lines(prefix, color, is_sorted=True, **kwargs):
    kwargs_items = sorted(kwargs.items()) if is_sorted else kwargs.items()
    for k, v in kwargs_items:
        cprint("{}{}: {}".format(prefix, k, v), color)


def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''    main_args = get_args(
        model_name="GAT",  # GAT, GCN
        dataset_class="Planetoid",  # Planetoid, FullPlanetoid, RandomPartitionGraph
        dataset_name="Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.1-0.025
        custom_key="EV13NSO8",  # NEO8, NEDPO8, EV13NSO8, EV9NSO8, EV1O8, EV2O8, -500, -Link, -ES, -ATT
    )'''


def get_args(model_name, yaml_path=None) -> argparse.Namespace:

    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")


    parser = argparse.ArgumentParser(description='Parser for Supervised Graph Attention Networks')

    # Basics
    parser.add_argument("--m", default="", type=str, help="Memo")
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--gpu-deny-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint-dir", default="../checkpoints")
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--task-type", default="", type=str)
    parser.add_argument("--perf-type", default="accuracy", type=str)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--verbose", default=2)
    parser.add_argument("--save-plot", default=False)
    parser.add_argument("--seed", default=42)

    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--data-sampling-size", default=None, type=int, nargs="+")
    parser.add_argument("--data-sampling-num-hops", default=None, type=int)
    parser.add_argument("--data-num-splits", default=1, type=int)
    parser.add_argument("--data-sampler", default=None, type=str)

    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--loss", default=None, type=str)
    parser.add_argument("--l1-lambda", default=0., type=float)
    parser.add_argument("--l2-lambda", default=0., type=float)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link

    # Early stop
    parser.add_argument("--use-early-stop", default=False, type=bool)
    parser.add_argument("--early-stop-patience", default=-1, type=int)
    parser.add_argument("--early-stop-queue-length", default=100, type=int)
    parser.add_argument("--early-stop-threshold-loss", default=-1.0, type=float)
    parser.add_argument("--early-stop-threshold-perf", default=-1.0, type=float)

    # Graph
    parser.add_argument("--num-hidden-features", default=64, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--out-heads", default=None, type=int)
    parser.add_argument("--pool-name", default=None)

    # Attention
    parser.add_argument("--is-super-gat", default=True, type=bool)
    parser.add_argument("--attention-type", default="basic", type=str)
    parser.add_argument("--att-lambda", default=0., type=float)
    parser.add_argument("--super-gat-criterion", default=None, type=str)
    parser.add_argument("--neg-sample-ratio", default=0.0, type=float)
    parser.add_argument("--scaling-factor", default=None, type=float)
    parser.add_argument("--to-undirected-at-neg", default=False, type=bool)
    parser.add_argument("--to-undirected", default=False, type=bool)

    # Pretraining
    parser.add_argument("--use-pretraining", default=False, type=bool)
    parser.add_argument("--total-pretraining-epoch", default=0, type=int)
    parser.add_argument("--pretraining-noise-ratio", default=0.0, type=float)

    # Baseline
    parser.add_argument("--is-link-gnn", default=False, type=bool)
    parser.add_argument("--link-lambda", default=0., type=float)

    parser.add_argument("--is-cgat-full", default=False, type=bool)
    parser.add_argument("--is-cgat-ssnc", default=False, type=bool)

    # Test
    parser.add_argument("--val-interval", default=10)

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args
