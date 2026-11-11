# -*- coding: utf-8 -*-
"""TransCDR CUDA assert bisect (temporary; delete after use).

Runs each op of DrugGCNEncoder separately on GPU with an explicit
torch.cuda.synchronize() after every step, so the failing kernel is pinned
to one line instead of surfacing later as a deferred launch error.

Usage (any dir):
    conda run -n bench-transcdr python <path to this file>
    # or, inside the env:
    python 4_DrugResponse_BCAHFP/benchmarks/transcdr/_diag_gcn.py
"""

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

print("torch          :", torch.__version__)
print("torch.cuda     :", torch.version.cuda)
print("cudnn          :", torch.backends.cudnn.version())
print("device         :", torch.cuda.get_device_name(0))
print("capability     :", torch.cuda.get_device_capability(0))

try:
    import torch_geometric
    print("torch_geometric:", torch_geometric.__version__)
except Exception as e:                                    # pragma: no cover
    print("torch_geometric: MISSING", e)

for _m in ("torch_scatter", "torch_sparse"):
    try:
        _mod = __import__(_m)
        print(f"{_m:15s}: {getattr(_mod, '__version__', '?')}")
    except Exception as e:
        print(f"{_m:15s}: MISSING ({type(e).__name__})")

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.utils import softmax as graph_softmax

from benchmarks.transcdr.train_transcdr import (WeightedSumAndMax, DrugGCNEncoder,
                                               GeneEncoder, TransCDRModel)

dev = "cuda"
FAILED = []


def step(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        if hasattr(out, "shape"):
            print(f"[OK]   {name:34s} -> {tuple(out.shape)}", flush=True)
        else:
            print(f"[OK]   {name:34s} -> {out}", flush=True)
        return out
    except Exception as e:
        FAILED.append(name)
        print(f"[FAIL] {name:34s} -> {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------- synthetic graphs
def make_graph(n_nodes):
    x = torch.randn(n_nodes, 78)
    src = torch.randint(0, n_nodes, (n_nodes * 2,))
    dst = torch.randint(0, n_nodes, (n_nodes * 2,))
    return Data(x=x, edge_index=torch.stack([src, dst]), num_nodes=n_nodes)


graphs = [make_graph(n) for n in (8, 5, 12, 3)]
batch = Batch.from_data_list(graphs).to(dev)


# ---------------------------------------------------------------- 1. primitives
def t_scatter_reduce():
    x = torch.randn(10, 4, device=dev)
    idx = torch.tensor([0] * 5 + [1] * 5, device=dev)
    return torch.scatter_reduce(x, 0, idx.unsqueeze(-1).expand(-1, 4), "amax",
                                include_self=False)


def t_batch_max_item():
    return batch.batch.max().item()


def t_global_max_pool():
    return global_max_pool(batch.x, batch.batch)


def t_softmax():
    return graph_softmax(torch.randn(batch.num_nodes, 1, device=dev), batch.batch, dim=0)


def t_index_add():
    w = torch.rand(batch.num_nodes, 64, device=dev)
    out = torch.zeros(batch.num_graphs, 64, device=dev)
    out.index_add_(0, batch.batch, w)
    return out


# ---------------------------------------------------------------- 2. GCN layers
conv1 = GCNConv(78, 64).to(dev)
conv2 = GCNConv(64, 64).to(dev)


def t_gcn1():
    return conv1(batch.x, batch.edge_index)


def t_gcn3():
    x = conv2(torch.relu(conv1(batch.x, batch.edge_index)), batch.edge_index)
    return torch.relu(x)


# ---------------------------------------------------------------- 3. readout / encoder
readout = WeightedSumAndMax(64).to(dev)


def t_readout():
    x = torch.relu(conv1(batch.x, batch.edge_index))
    return readout(x, batch.batch)


enc = DrugGCNEncoder(out_dim=256, dropout=0.1).to(dev)


def t_drug_encoder():
    return enc(batch)


gen = GeneEncoder(256, out_dim=256, dropout=0.1).to(dev)
gene = torch.randn(batch.num_graphs, 256, device=dev)


def t_gene_encoder():
    return gen(gene)


model = TransCDRModel(256, dropout=0.1).to(dev)


def t_full():
    out = model(batch, gene)
    loss = out.pow(2).mean()
    loss.backward()
    return out


step("scatter_reduce(amax)", t_scatter_reduce)
step("batch.batch.max().item()", t_batch_max_item)
step("global_max_pool", t_global_max_pool)
step("graph_softmax", t_softmax)
step("index_add_", t_index_add)
step("GCNConv(1 layer)", t_gcn1)
step("GCNConv(2 layers)", t_gcn3)
step("WeightedSumAndMax", t_readout)
step("DrugGCNEncoder", t_drug_encoder)
step("GeneEncoder", t_gene_encoder)
step("TransCDRModel fwd+bwd", t_full)

print("=" * 60)
if FAILED:
    print("FAILED STEPS:", ", ".join(FAILED))
else:
    print("ALL STEPS OK (synthetic graphs) -> the crash needs real data;")
    print("rerun training with CUDA_LAUNCH_BLOCKING=1 to pin the real op.")
