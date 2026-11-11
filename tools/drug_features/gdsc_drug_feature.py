# -*- coding: utf-8 -*-
"""
extract FG-BERT atom features from the SMILES in the GDSC response table, output HDF5.
"""

import os
import sys
import random
import hashlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from tqdm import tqdm

# fixed random seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from utils import smiles2adjoin
    from dataset_scaffold_random import str2num
except ImportError:
    raise ImportError("make sure utils.py and dataset_scaffold_random.py are in the current directory")


def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


# ==================== 1. transformer components ====================
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None, adjoin_matrix=None):
        """padding mask and adjoin are two separate channels."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        if adjoin_matrix is not None:
            scaled_attention_logits += adjoin_matrix
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, v, k, q, mask=None, adjoin_matrix=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(PointWiseFeedForwardNetwork, self).__init__()
        # FFN uses gelu
        self.dense1 = tf.keras.layers.Dense(dff, activation=gelu)
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff, rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None, adjoin_matrix=None, training=False):
        # expand adjoin and pass it to attention separately
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        attn_output, attention_weights = self.mha(x, x, x, mask=mask,
                                                  adjoin_matrix=adjoin_matrix)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2, attention_weights


class BertModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate=0.1):
        super(BertModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, adjoin_matrix=None, training=False):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x, _ = layer(x, mask, adjoin_matrix=adjoin_matrix, training=training)
        return x


# ==================== 2. weight loading ====================
def _tail(path):
    """take the last two path segments of a variable as its key."""
    return '/'.join(path.replace(':0', '').split('/')[-2:])


def load_fgbert_weights(model, weights_path, strict=True):
    """load pretrained weights matched by layer-name suffix."""
    import h5py
    with h5py.File(weights_path, 'r') as f:
        stored = {}
        f.visititems(
            lambda n, o: stored.__setitem__(n, np.asarray(o[()]))
            if isinstance(o, h5py.Dataset) else None
        )

    pool = {}
    for k in stored:
        pool.setdefault(_tail(k), []).append(k)

    ok, missing = 0, []
    for v in model.weights:
        name = getattr(v, 'path', v.name)
        cands = [c for c in pool.get(_tail(name), []) if stored[c].shape == tuple(v.shape)]
        if cands:
            v.assign(stored[cands[0]])
            ok += 1
        else:
            missing.append(name)

    total = len(model.weights)
    print(f"weight loading: {ok}/{total} matched ({len(stored)} arrays in the weights file)")
    if missing:
        for m in missing[:20]:
            print("   unmatched:", m)
    if strict and ok != total:
        raise RuntimeError(f"FG-BERT weights not fully loaded ({ok}/{total})")


# ==================== 3. feature extractor ====================
class DualFeatureExtractor:
    def __init__(self, encoder_weights_path, use_adjoin=True):
        self.num_layers = 6
        self.d_model = 256
        self.num_heads = 8
        self.dff = self.d_model * 2
        self.vocab_size = 18
        self.addH = True
        # use_adjoin=False turns off the bond constraint and falls back to full attention
        self.use_adjoin = use_adjoin

        self.bert = BertModel(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            vocab_size=self.vocab_size,
        )
        # run once to build the variables
        dummy_x = tf.constant([[1, 2, 3, 0, 0]], dtype=tf.int64)
        dummy_mask = tf.constant([[[[1.0, 1.0, 1.0, 0.0, 0.0]]]], dtype=tf.float32)
        _ = self.bert(dummy_x, dummy_mask, adjoin_matrix=None, training=False)

        # load pretrained weights
        load_fgbert_weights(self.bert, encoder_weights_path, strict=True)
        self.bert.trainable = False

    def preprocess_smiles(self, smiles):
        """returns (token_ids, adjoin_mask_tensor, padding_mask)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid SMILES")
        smiles_std = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        atoms_list, adj_orig = smiles2adjoin(smiles_std, explicit_hydrogens=self.addH)
        if not atoms_list:
            raise ValueError("empty atom list")
        atoms_list = ['<global>'] + atoms_list
        num_nodes = len(atoms_list)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adj_matrix[0, 1:] = 1.0
        adj_matrix[1:, 0] = 1.0
        adj_matrix[1:, 1:] = adj_orig
        adjoin_mask = np.full((num_nodes, num_nodes), -1e9, dtype=np.float32)
        adjoin_mask[adj_matrix == 1] = 0.0
        nums_list = [str2num.get(atom, str2num['<unk>']) for atom in atoms_list]
        seq = tf.cast(tf.math.equal(nums_list, 0), tf.float32)
        mask = tf.expand_dims(seq, axis=0)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=1)
        x = tf.convert_to_tensor([nums_list], dtype=tf.int64)
        adjoin_mask_tensor = tf.convert_to_tensor([adjoin_mask], dtype=tf.float32)
        return x, adjoin_mask_tensor, mask

    def extract_features(self, smiles_list):
        atom_feats, global_feats = [], []
        print(f"extracting features for {len(smiles_list)} molecules")
        for smi in tqdm(smiles_list, desc="Processing"):
            try:
                x, adjoin_mask, mask = self.preprocess_smiles(smi)
                adj = adjoin_mask if self.use_adjoin else None
                enc_out = self.bert(x, mask, adjoin_matrix=adj, training=False).numpy().squeeze(0)
                atom_feats.append(enc_out[1:, :])   # drop the global node
                global_feats.append(enc_out[0, :])
            except Exception as e:
                print(f"failed on SMILES: {str(smi)[:50]}... error: {e}")
                atom_feats.append(np.zeros((0, self.d_model), dtype=np.float32))
                global_feats.append(np.zeros(self.d_model, dtype=np.float32))
        return atom_feats, global_feats


# ==================== 4. main flow ====================
def clean_smiles(smiles_str):
    if not isinstance(smiles_str, str):
        return None
    parts = smiles_str.split(',')
    for part in parts:
        part = part.strip()
        if part:
            try:
                mol = Chem.MolFromSmiles(part)
                if mol is not None:
                    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            except Exception:
                continue
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description='extract drug atom features from the GDSC response table')
    ap.add_argument('--data_csv', type=str, default='../Data/Drug/GDSC_RES.csv',
                    help='GDSC response table CSV path')
    ap.add_argument('--weights_h5', type=str, default='fg_bert_weights.h5',
                    help='FG-BERT pretrained weights path')
    ap.add_argument('--out_h5', type=str, default='../Data/Drug/GDSC_RES_FGBERT.h5',
                    help='output h5 path')
    ap.add_argument('--no_adjoin', action='store_true',
                    help='disable the bond-constrained attention')
    args = ap.parse_args()

    data_csv = args.data_csv
    weights_h5 = args.weights_h5
    output_h5 = args.out_h5
    output_global_pkl = output_h5.replace('.h5', '_global.pkl')

    print(f"output: {output_h5}")
    print(f"attention: {'bond-constrained (adjoin)' if not args.no_adjoin else 'full (no adjoin)'}")
    print("reading raw data...")
    df = pd.read_csv(data_csv)
    required_cols = ['DRUG_ID', 'DRUG_SMILE', 'GDSC_ID', 'LN_IC50']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    drug_smiles_map = {}
    for drug_id, group in df.groupby('DRUG_ID'):
        valid_smiles = None
        for smi in group['DRUG_SMILE']:
            cleaned = clean_smiles(smi)
            if cleaned:
                valid_smiles = cleaned
                break
        if valid_smiles:
            drug_smiles_map[drug_id] = valid_smiles
        else:
            print(f"warning: drug {drug_id} has no valid SMILES, skipped")

    drug_ids = list(drug_smiles_map.keys())
    unique_smiles = [drug_smiles_map[did] for did in drug_ids]
    print(f"valid drugs: {len(drug_ids)}")

    extractor = DualFeatureExtractor(encoder_weights_path=weights_h5,
                                     use_adjoin=not args.no_adjoin)
    atom_feats_list, global_feats_list = extractor.extract_features(unique_smiles)

    with pd.HDFStore(output_h5, mode='w') as store:
        drug_smiles_df = pd.DataFrame({'DRUG_ID': drug_ids, 'DRUG_SMILE': unique_smiles})
        store.put('drug_smiles', drug_smiles_df)
        atom_series = pd.Series(atom_feats_list, name='atom_features')
        store.put('atom_features', atom_series.to_frame())
        store.put('original_data', df)
    print(f"atom features saved to: {output_h5}")

    global_dict = {str(did): gf for did, gf in zip(drug_ids, global_feats_list)}
    with open(output_global_pkl, 'wb') as f:
        pickle.dump(global_dict, f)
    print(f"global features saved to: {output_global_pkl}")

    # md5 fingerprint of atom_features
    h = hashlib.md5()
    for a in atom_feats_list:
        h.update(np.asarray(a, dtype=np.float32).tobytes())
    print("=" * 60)
    print(f"atom_features md5 = {h.hexdigest()}")
    print(f"drugs = {len(drug_ids)}   raw samples = {len(df)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
