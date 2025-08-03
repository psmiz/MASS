# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os

import tqdm

from domainbed.lib.query import Q

def load_records(path, seed=None):
    records = []
    for _, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        if seed is None or (seed is not None and subdir[-1] == str(seed)):
            results_path = os.path.join(path, subdir, "results.jsonl")
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass
    return Q(records)

def load_records_data(path, seed=None):
    records = []
    for _, subdir1 in list(enumerate(os.listdir(path))):
        # import pdb;pdb.set_trace()
        if subdir1 == "results.txt" or subdir1 == "results.tex" or ".log" in subdir1: continue
        path1 = os.path.join(path, subdir1, "GMOE")
        for j, subdir2 in list(enumerate(os.listdir(path1))):
            if subdir2 == "results.txt" or subdir2 == "results.tex" or ".log" in subdir2: continue
            if seed is None or (seed is not None and subdir2.split('_')[-1] == str(seed)): 
                results_path = os.path.join(path1, subdir2, "results.jsonl")
                try:
                    with open(results_path, "r") as f:
                        for line in f:
                            records.append(json.loads(line[:-1]))
                except IOError:
                    pass

    return Q(records)

def load_records_all(path, seed=None):
    records = []
    for _, subdir1 in list(enumerate(os.listdir(path))):
        if subdir1 == "results.txt" or subdir1 == "results.tex" or ".log" in subdir1: continue
        path1 = os.path.join(path, subdir1)
        for _, subdir2 in list(enumerate(os.listdir(path1))):
            if subdir2 == "results.txt" or subdir2 == "results.tex" or ".log" in subdir2: continue
            path2 = os.path.join(path1, subdir2, "GMOE")
            for j, subdir3 in list(enumerate(os.listdir(path2))):
                if subdir3 == "results.txt" or subdir3 == "results.tex" or ".log" in subdir3: continue
                if seed is None or (seed is not None and subdir3.split('_')[-1] == str(seed)): 
                    results_path = os.path.join(path2, subdir3, "results.jsonl")
                    try:
                        with open(results_path, "r") as f:
                            for line in f:
                                records.append(json.loads(line[:-1]))
                    except IOError:
                        pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]:
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env)
            result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()])
