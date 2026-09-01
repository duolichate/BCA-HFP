# bca_hfp/data/dataset.py
import torch
from torch.utils.data import Dataset


# For cross-attention model
class DrugGeneDataset(Dataset):
    # Takes drug IDs, cell IDs, labels, and feature dicts. Keeps only samples with valid IDs in all dicts.
    def __init__(self, drug_ids, cell_ids, labels, drug_atom_dict, gene_emb_dict, drug_global_dict):
        self.samples = []
        for did, cid, lab in zip(drug_ids, cell_ids, labels):
            if did not in drug_atom_dict:
                raise KeyError(f"Drug {did} missing atom features")
            if cid not in gene_emb_dict:
                raise KeyError(f"Cell {cid} missing gene embeddings")
            if did not in drug_global_dict:
                raise KeyError(f"Drug {did} missing global features")
            global_feat = drug_global_dict[did]
            self.samples.append((drug_atom_dict[did], gene_emb_dict[cid], lab, global_feat))
        print(f"Valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # Returns (atom_feat, gene_emb, label) as torch tensors
    def __getitem__(self, idx):
        atom_feat, gene_emb, label, global_feat = self.samples[idx]
        return (torch.tensor(atom_feat, dtype=torch.float32),
                torch.tensor(gene_emb, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(global_feat, dtype=torch.float32))


# Collate function for DrugGeneDataset
def collate_fn(batch, atom_mask_prob=0.0):
    """
    Pad atom features to max length in batch. Apply random masking if atom_mask_prob > 0.
    Stack gene features (fixed length). Returns (padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats).
    """
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
    return padded_atoms, atom_masks, gene_embs, gene_masks, labels, global_feats


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

