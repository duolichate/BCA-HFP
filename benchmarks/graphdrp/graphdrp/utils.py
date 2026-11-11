#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphDRP adaptation utilities: dataset class and evaluation metrics.
"""

import os
import torch
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import numpy as np


class GraphDRPDataset(InMemoryDataset):
    """
    GraphDRP dataset adapted for custom data.
    Drug: PyG graph (atom features + edges)
    Cell line: 256-dim Geneformer embedding
    Label: LN_IC50
    """

    def __init__(self, root, drug_graphs, cell_features, labels, transform=None, pre_transform=None):
        self.drug_graphs = drug_graphs
        self.cell_features = cell_features
        self.labels = labels
        super(GraphDRPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for i in range(len(self.labels)):
            graph = self.drug_graphs[i]
            cell_feat = torch.tensor(self.cell_features[i], dtype=torch.float32)
            label = torch.tensor([self.labels[i]], dtype=torch.float32)

            gdata = graph.clone()
            gdata.y = label
            gdata.target = cell_feat
            if not hasattr(gdata, 'num_nodes') or gdata.num_nodes is None:
                gdata.num_nodes = gdata.x.size(0)
            gdata.edge_index = gdata.edge_index.long()
            data_list.append(gdata)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    return np.sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    from scipy.stats import spearmanr
    return spearmanr(y, f)[0]