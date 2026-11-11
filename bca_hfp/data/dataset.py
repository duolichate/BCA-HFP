# bca_hfp/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# For cross-attention model
class DrugGeneDataset(Dataset):
    # Takes drug IDs, cell IDs, labels, and feature dicts. Keeps only samples with valid IDs in all dicts.
    # drug_to_idx: optional {drug_id: int}; when given, __getitem__ returns it as well
    #               for drug-grouped losses (see DrugGroupedSampler).
    def __init__(self, drug_ids, cell_ids, labels, drug_atom_dict, gene_emb_dict, drug_global_dict,
                 drug_to_idx=None, prior_dict=None, cens_arr=None):
        self.samples = []
        self.drug_ids = []
        self.prior_dict = prior_dict
        self.prior_dim = 0
        if prior_dict:
            for v in prior_dict.values():
                if v is not None:
                    self.prior_dim = int(np.asarray(v).shape[0])
                    break
        for did, cid, lab in zip(drug_ids, cell_ids, labels):
            if did not in drug_atom_dict:
                raise KeyError(f"Drug {did} missing atom features")
            if cid not in gene_emb_dict:
                raise KeyError(f"Cell {cid} missing gene embeddings")
            if did not in drug_global_dict:
                raise KeyError(f"Drug {did} missing global features")
            global_feat = drug_global_dict[did]
            self.samples.append((drug_atom_dict[did], gene_emb_dict[cid], lab, global_feat))
            self.drug_ids.append(did)
        self.drug_to_idx = drug_to_idx
        # censoring info: (N, 4) = [b_lo, b_hi, cens_flag, w], row-aligned with drug_ids
        # cens_flag: +1 right-censored (true value >= b_hi), -1 left-censored (<= b_lo), 0 inside window
        self.cens_arr = None
        if cens_arr is not None:
            arr = np.asarray(cens_arr, dtype=np.float32)
            if len(arr) != len(self.samples):
                raise ValueError(f"cens_arr length {len(arr)} != number of samples {len(self.samples)}")
            self.cens_arr = arr
        print(f"Valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # Returns (atom_feat, gene_emb, label, global_feat[, drug_idx][, prior][, cens]) as torch tensors
    def __getitem__(self, idx):
        atom_feat, gene_emb, label, global_feat = self.samples[idx]
        out = (torch.tensor(atom_feat, dtype=torch.float32),
               torch.tensor(gene_emb, dtype=torch.float32),
               torch.tensor(label, dtype=torch.float32),
               torch.tensor(global_feat, dtype=torch.float32))
        if self.drug_to_idx is not None:
            out = out + (torch.tensor(self.drug_to_idx[self.drug_ids[idx]], dtype=torch.long),)
        if self.prior_dict is not None:
            pv = self.prior_dict.get(self.drug_ids[idx])
            if pv is None:
                pv = np.zeros((max(self.prior_dim, 1),), dtype=np.float32)
            out = out + (torch.tensor(np.asarray(pv), dtype=torch.float32),)
        # cens always goes last (collate_fn strips it under the same convention)
        if self.cens_arr is not None:
            out = out + (torch.from_numpy(self.cens_arr[idx].copy()),)
        return out


# Collate function for DrugGeneDataset
def collate_fn(batch, atom_mask_prob=0.0, with_drug_idx=False, with_prior=False,
               with_cens=False):
    """
    Pad atom features to max length in batch. Apply random masking if atom_mask_prob > 0.
    Stack gene features (fixed length).
    Returns (padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats)
    and, when with_drug_idx=True, an extra trailing drug_idx tensor of shape (B,),
    and, when with_prior=True, an extra trailing prior tensor of shape (B, prior_dim),
    and, when with_cens=True, an extra trailing cens tensor of shape (B, 4).
    trailing tensor order: [drug_idx], [prior], [cens]
    """
    cens = None
    if with_cens:
        # __getitem__ appends cens last; strip it before the usual unpacking
        cens = torch.stack([b[-1] for b in batch])
        batch = tuple(b[:-1] for b in batch)

    if with_prior:
        if with_drug_idx:
            atom_feats, gene_embs, labels, global_feats, drug_idxs, priors = zip(*batch)
        else:
            atom_feats, gene_embs, labels, global_feats, priors = zip(*batch)
    elif with_drug_idx:
        atom_feats, gene_embs, labels, global_feats, drug_idxs = zip(*batch)
    else:
        atom_feats, gene_embs, labels, global_feats = zip(*batch)

    # Pad atom features
    atom_lengths = [f.shape[0] for f in atom_feats]
    max_atoms = max(atom_lengths)
    padded_atoms = []
    atom_masks = []
    for feat in atom_feats:
        pad_len = max_atoms - feat.shape[0]
        padded = torch.cat([feat, torch.zeros(pad_len, feat.shape[1])], dim=0)
        padded_atoms.append(padded)
        mask = torch.ones(feat.shape[0], dtype=torch.bool)
        mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)])
        atom_masks.append(mask)
    padded_atoms = torch.stack(padded_atoms)  # (B, max_atoms, 256)
    atom_masks = torch.stack(atom_masks)  # (B, max_atoms)

    # Random atom masking
    if atom_mask_prob > 0:
        random_mask = torch.rand_like(padded_atoms) < atom_mask_prob
        padded_atoms[random_mask] = 0.0

    # Gene features
    gene_embs = torch.stack(gene_embs)  # (B, n_genes, dim)
    gene_masks = torch.ones(gene_embs.shape[0], gene_embs.shape[1], dtype=torch.bool)

    labels = torch.tensor(labels, dtype=torch.float32)

    global_feats = torch.stack(global_feats)
    if with_prior and with_drug_idx:
        out = (padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats,
               torch.stack(drug_idxs), torch.stack(priors))
    elif with_prior:
        out = (padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats,
               torch.stack(priors))
    elif with_drug_idx:
        out = (padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats,
               torch.stack(drug_idxs))
    else:
        out = (padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats)
    if with_cens:
        out = out + (cens,)
    return out


class DrugGroupedBatchSampler(Sampler):
    """drug-grouped batch sampler: yields one list of indices per batch.

    Use it as DataLoader(batch_sampler=...), not sampler=..., since each yield
    is a list of indices for one batch rather than a single sample index.

    Motivation: with plain shuffle at batch=32 over a few dozen drugs, a drug
    usually has 0-1 samples per batch, so the drug-mean auxiliary loss collapses
    to per-sample L1. Grouping by drug gives each drug samples_per_drug samples
    per batch, which makes the drug mean meaningful.

    Steps per epoch are fixed at len(dataset) // batch_size, same as plain
    shuffle, so epoch length stays comparable. Sampling is with replacement, so
    an epoch covers only part of the data (unlike plain shuffle), but steps per
    epoch and samples per epoch match.
    """

    def __init__(self, drug_ids, batch_size, drugs_per_batch=8, seed=42):
        self.drug_to_pos = {}
        for i, d in enumerate(drug_ids):
            self.drug_to_pos.setdefault(d, []).append(i)
        self.unique_drugs = list(self.drug_to_pos)
        self.drugs_per_batch = min(int(drugs_per_batch), len(self.unique_drugs))
        self.batch_size = int(batch_size)
        self.samples_per_drug = max(1, self.batch_size // self.drugs_per_batch)
        self.steps = max(1, len(drug_ids) // self.batch_size)
        self.seed = int(seed)
        self._call = 0

    def __iter__(self):
        # bump the internal counter so each epoch draws a different but reproducible sequence
        rng = np.random.default_rng(self.seed + self._call)
        self._call += 1
        for _ in range(self.steps):
            picked_drugs = rng.choice(len(self.unique_drugs), size=self.drugs_per_batch, replace=False)
            batch = []
            for j in picked_drugs:
                pool = self.drug_to_pos[self.unique_drugs[int(j)]]
                replace = len(pool) < self.samples_per_drug
                picked = rng.choice(pool, size=self.samples_per_drug, replace=replace)
                batch.extend(int(i) for i in picked)
            yield batch

    def __len__(self):
        return self.steps


class MixedBatchSampler(Sampler):
    """mixed sampling: within each group_every batches, the first group_every-1 are plain
    random and the group_every-th is drug-grouped.

    Motivation: plain shuffle is the standard protocol (uniform over samples), but
    a batch usually holds 0-1 samples of a given drug, so the drug-mean auxiliary
    loss decays to 0. A 3:1 mix keeps the standard random body and spends 1/4 of
    the steps on a same-drug multi-sample batch that makes the aux loss meaningful.
    """

    def __init__(self, n_samples, drug_ids, batch_size, drugs_per_batch=8,
                 group_every=4, seed=42):
        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.group_every = max(2, int(group_every))
        self.steps = max(1, self.n // self.batch_size)
        self.grouped = DrugGroupedBatchSampler(drug_ids, batch_size, drugs_per_batch, seed)
        self.seed = int(seed)
        self._call = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 10_000 + self._call)
        self._call += 1
        perm = rng.permutation(self.n)          # plain random part: every sample appears once per epoch
        pos = 0
        grouped_iter = iter(self.grouped)
        for step in range(self.steps):
            if (step + 1) % self.group_every == 0:
                try:
                    yield next(grouped_iter)
                except StopIteration:
                    grouped_iter = iter(self.grouped)
                    yield next(grouped_iter)
                continue
            take = perm[pos:pos + self.batch_size]
            pos += self.batch_size
            if len(take) < self.batch_size:      # should not happen (steps = n // bs)
                take = np.concatenate([take, rng.integers(0, self.n, self.batch_size - len(take))])
            yield take.tolist()

    def __len__(self):
        return self.steps


# For baseline model
class BaselineDataset(Dataset):
    # Takes pooled drug/gene feature dicts and generates sample list.
    def __init__(self, drug_ids, cell_ids, labels, drug_pooled, gene_pooled):
        self.samples = []
        for did, cid, lab in zip(drug_ids, cell_ids, labels):
            if did in drug_pooled and cid in gene_pooled:
                self.samples.append((drug_pooled[did], gene_pooled[cid], lab))
        print(f"Valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # Returns (drug_vec, gene_vec, label)
    def __getitem__(self, idx):
        drug, gene, label = self.samples[idx]
        return torch.tensor(drug, dtype=torch.float32), torch.tensor(gene, dtype=torch.float32), torch.tensor(label,
                                                                                                              dtype=torch.float32)
