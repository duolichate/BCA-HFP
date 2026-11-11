# bca_hfp/data/interaction_prior.py
"""interaction prior: compress the cross-drug extrapolable drug x cell interaction term into a K-dim vector.

Background:
  under GDSC2 LDO, splitting the prediction as
      y ~ mu + a_d + b_c + interaction
  gives ~64.5% of the within-drug signal from the cell-line mean sensitivity
  b_c (one table lookup); the remaining interaction term is weak but does
  extrapolate across drugs from fingerprints:
      two-stage estimator = b_c + alpha * dev_pred
      within R2 0.3582 -> 0.3885, within rho 0.5993 -> 0.6237 (5-fold mean)
  the network at that time scored 0.3192 / 0.5805, losing to this trivial
  baseline within drug.

This module does one thing: for each drug output a K-dim interaction
coefficient c_d such that
      dev_pred[d, :] = c_d @ V_k
  reconstructs the within-drug curve shape over 941 cell lines.
On the model side it is a zero-initialised extra input (prior_proj in
models/main_model.py); training-drug coefficients use leave-one-drug-out
predictions (same distribution as test drugs), which avoids the shift of
"training drugs get a perfect prior, test drugs get an extrapolated one".

Only training-fold (drug, cell, label) triples are used; val/test labels never.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# basic math
# --------------------------------------------------------------------------- #


def _two_way(di, ci, y, n_d, n_c, iters=30, lam=5.0):
    """ALS fit of y ~ mu + a_d + b_c (drug + cell-line additive effects, shrunken)."""
    mu = float(np.mean(y))
    r = y - mu
    a = np.zeros(n_d, dtype=np.float64)
    b = np.zeros(n_c, dtype=np.float64)
    cnt_d = np.maximum(np.bincount(di, minlength=n_d).astype(np.float64), 1.0)
    cnt_c = np.maximum(np.bincount(ci, minlength=n_c).astype(np.float64), 1.0)
    shrink_d = 1.0 / (1.0 + lam / cnt_d.mean())
    shrink_c = 1.0 / (1.0 + lam / cnt_c.mean())
    for _ in range(iters):
        num_c = np.bincount(ci, weights=r - a[di], minlength=n_c)
        b_new = (num_c / cnt_c) * shrink_c
        nb = np.bincount(di, weights=b_new[ci], minlength=n_d)
        a_new = ((np.bincount(di, weights=r, minlength=n_d) - nb) / cnt_d) * shrink_d
        delta = float(np.abs(b_new - b).max() + np.abs(a_new - a).max())
        a, b = a_new, b_new
        if delta < 1e-6:
            break
    return mu, a, b


def _normalize_fp(X):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    scale = float(np.mean(np.sqrt((Xc * Xc).sum(axis=1)))) or 1.0
    return Xc / scale, mean, scale


def _ridge_multi(Xc, C, lam):
    """multi-output dual-form ridge (X already centered and scaled).

    regularisation: lam times the mean squared sample norm, mean_i ||x_i||^2 = trace(A)/n.
    Do not use trace(A)/p here: p is the feature dim (2048) and n the sample count (~182),
    an order of magnitude apart, which would put the effective lam off by 10x.
    """
    n, p = Xc.shape
    A = Xc.T @ Xc
    reg = lam * float(np.trace(A)) / max(n, 1)
    A = A + reg * np.eye(p, dtype=A.dtype)
    B = Xc.T @ C
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, B, rcond=None)[0]
    return W.reshape(p, C.shape[1])


def _cos_rows(A, B):
    a = A.ravel()
    b = B.ravel()
    na = np.sqrt(float((a * a).sum()))
    nb = np.sqrt(float((b * b).sum()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #


def read_smiles_map(h5_path):
    """DRUG_ID -> SMILES (handles both pandas HDFStore and h5py group layouts)."""
    try:
        store = pd.HDFStore(h5_path, mode="r")
        try:
            key = "/drug_smiles" if "/drug_smiles" in store.keys() else "drug_smiles"
            df = store[key]
        finally:
            store.close()
    except Exception:
        import h5py

        with h5py.File(h5_path, "r") as f:
            grp = f["drug_smiles"]
            df = pd.DataFrame({c: np.asarray(grp[c][:]) for c in grp.keys()})

    smi_col = None
    for c in df.columns:
        if "SMILE" in str(c).upper():
            smi_col = c
            break
    if smi_col is None:
        raise RuntimeError(f"no SMILES column in the drug_smiles table: {list(df.columns)}")
    out = {}
    for k, v in zip(df["DRUG_ID"], df[smi_col]):
        if v is None:
            continue
        s = str(v).strip()
        if s in ("", "None", "nan"):
            continue
        out[str(k)] = s
    return out


def build_fps(smiles_map, n_bits=2048, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    RDLogger.DisableLog("rdApp.*")
    out = {}
    for did, smi in smiles_map.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        out[did] = arr
    return out


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def fit_interaction_prior(pairs, h5_path, label_col, drug_col, cell_col, K=16,
                          als_lam=5.0, als_iters=30,
                          lam_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
                          inner_splits=4, seed=0, fp_bits=2048, fp_radius=2,
                          fp_cache=None, verbose=True):
    """compute K-dim interaction coefficients per drug from training-fold data.

    Parameters
    ----------
    pairs : DataFrame with training-fold triples only (drug_col, cell_col, label_col)
    h5_path : h5 holding the drug_smiles table (for fingerprints)
    K : SVD basis dimension (16 worked best, 8/32 comparable)
    fp_cache : optional {drug_id: fp} cache reused across folds to save time

    Returns
    -------
    prior_dict : {drug_id: np.ndarray(K,)} -- training drugs use LOGO predictions
                 (same distribution as test drugs); other drugs with a
                 fingerprint use full-fit ridge extrapolation; drugs without a
                 fingerprint get a zero vector.
    info : dict(lam, cos, n_drugs, n_cells, n_covered)
    """
    df = pairs[[drug_col, cell_col, label_col]].dropna()
    if len(df) == 0:
        raise RuntimeError("[prior] training fold is empty")
    drug_ids = sorted(df[drug_col].astype(str).unique().tolist())
    cell_ids = sorted(df[cell_col].astype(str).unique().tolist())
    d_code = {d: i for i, d in enumerate(drug_ids)}
    c_code = {c: i for i, c in enumerate(cell_ids)}
    di = df[drug_col].astype(str).map(d_code).values
    ci = df[cell_col].astype(str).map(c_code).values
    y = df[label_col].values.astype(np.float64)
    n_d, n_c = len(drug_ids), len(cell_ids)

    mu, a, b = _two_way(di, ci, y, n_d, n_c, iters=als_iters, lam=als_lam)
    s = y - mu - b[ci]

    # within-drug curve matrix of training drugs (spread over cell lines, missing 0, centered per drug)
    D = np.zeros((n_d, n_c), dtype=np.float64)
    W = np.zeros((n_d, n_c), dtype=np.float64)
    np.add.at(D, (di, ci), s)
    np.add.at(W, (di, ci), 1.0)
    dev = np.where(W > 0, D / np.maximum(W, 1e-9), 0.0)
    wsum = W.sum(axis=1, keepdims=True)
    dev = dev - (dev * W).sum(axis=1, keepdims=True) / np.maximum(wsum, 1e-9)

    U, S, Vt = np.linalg.svd(dev, full_matrices=False)
    K = int(min(K, S.shape[0]))
    C = U[:, :K] * S[:K]                       # (n_d, K) true coefficients of training drugs

    # fingerprints
    fps = fp_cache
    if fps is None:
        fps = build_fps(read_smiles_map(h5_path), n_bits=fp_bits, radius=fp_radius)
    have = [i for i, d in enumerate(drug_ids) if d in fps]
    prior = {d: np.zeros((K,), dtype=np.float32) for d in drug_ids}
    if len(have) < max(10, K + 2):
        if verbose:
            print(f"[prior][warn] only {len(have)} training drugs have a fingerprint, prior set to zero")
        return prior, dict(lam=float("nan"), cos=0.0, n_drugs=n_d,
                           n_cells=n_c, n_covered=len(have))

    idx = np.array(have)
    X = np.stack([fps[drug_ids[i]] for i in idx], axis=0)
    Xc, fp_mean, fp_scale = _normalize_fp(X)
    Cx = C[idx]

    # inner lambda selection: split training drugs into inner_splits groups, compare cosine after held-out prediction
    rng = np.random.RandomState(seed)
    groups = rng.randint(0, inner_splits, size=len(idx))
    best = (-1.0, lam_grid[0])
    for lam in lam_grid:
        cos_list = []
        for g in range(inner_splits):
            m = groups != g
            if m.sum() < 10 or (~m).sum() == 0:
                continue
            Wg = _ridge_multi(Xc[m], Cx[m], lam)
            cos_list.append(_cos_rows(Xc[~m] @ Wg, Cx[~m]))
        score = float(np.mean(cos_list)) if cos_list else -1.0
        if score > best[0]:
            best = (score, lam)
    lam = float(best[1])
    cos = float(best[0])

    # training drugs: leave-one-group-out prediction (avoids inflated in-sample amplitude)
    P_tr = np.zeros((len(idx), K), dtype=np.float64)
    for g in range(inner_splits):
        m = groups != g
        if m.sum() < 10 or (~m).sum() == 0:
            continue
        Wg = _ridge_multi(Xc[m], Cx[m], lam)
        P_tr[~m] = Xc[~m] @ Wg

    # other drugs (val/test): extrapolate with the full-fit ridge; reuse the training-drug normalisation
    Wfull = _ridge_multi(Xc, Cx, lam)
    in_train = set(drug_ids)
    P_extra = {}
    for d, fp in fps.items():
        if d in in_train:
            continue
        x = (fp[None, :] - fp_mean) / fp_scale
        P_extra[d] = (x @ Wfull)[0]

    # per-dim standardisation using training-drug statistics only; SVD coefficients
    # carry a basis-dependent scale (some dims differ by >10x in std), and without
    # this the prior is either crushed to noise or dominates the fusion vector.
    # Linear(16,256) can relearn any scale, so nothing is lost here.
    mu_k = P_tr.mean(axis=0)
    sd_k = P_tr.std(axis=0)
    sd_k = np.where(sd_k < 1e-9, 1.0, sd_k)
    for j, i in enumerate(idx):
        prior[drug_ids[i]] = ((P_tr[j] - mu_k) / sd_k).astype(np.float32)
    for d, v in P_extra.items():
        prior[d] = ((v - mu_k) / sd_k).astype(np.float32)
    n_extra = len(P_extra)

    if verbose:
        print(f"[prior] n_drugs={n_d} n_cells={n_c} K={K} lam={lam:.3f} "
              f"inner_cos={cos:+.4f} train_drugs={len(idx)} extra_drugs={n_extra} "
              f"prior_sd={float(np.mean(sd_k)):.3f}")
    return prior, dict(lam=lam, cos=cos, n_drugs=n_d, n_cells=n_c,
                       n_covered=len(idx) + n_extra, mu_k=mu_k, sd_k=sd_k)
