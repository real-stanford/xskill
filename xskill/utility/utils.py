import os
import pickle
from typing import Any, Callable, Dict, Optional

from absl import logging
from torch import nn
import json


def read_json(path: str):
    """Read a JSON file."""
    with open(path, "r") as file:
        data = json.load(file)
    return data


def write_json(path: str, data: Any):
    with open(path, "w") as file:
        json.dump(data, file)


def replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module],
                                                                   bool],
                       func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    bn_list = [
        k.split('.')
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split('.')
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def save_pickle(experiment_path, arr, name):
    """Save an array as a pickle file."""
    filename = os.path.join(experiment_path, name)
    with open(filename, "wb") as fp:
        pickle.dump(arr, fp)
    logging.info("Saved %s to %s", name, filename)


def load_pickle(pretrained_path, name):
    """Load a pickled array."""
    filename = os.path.join(pretrained_path, name)
    with open(filename, "rb") as fp:
        arr = pickle.load(fp)
    logging.info("Successfully loaded %s from %s", name, filename)
    return arr
