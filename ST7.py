#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST6.py — Spatial Transcriptomics + H&E → SNO4-style Spectral Encodings (+ Slepian + Fractal) → Multi-gene Regression

This script CONSERVES ST5.py behavior and extends it with:
  • Graph Slepians (localized spectral basis) with k-means region masks
  • Fixed Spatial Autocorrelation weights (Gaussian ε-graph by default; KNN optional)
  • All previous spectral methods kept: HKS, Graph Wave Energy, Persistent Topology, Fractal Encoder

Feature groups (toggle with flags):
  • SNO4 patch morphology (8 ch) — always on
  • Heat Kernel Signature (HKS) on Laplacian eigenbasis (per-spot)
  • Graph Wave Energy (GWE) via spectral graph filters (per-spot)
  • Persistent Topology on patches via Gudhi (per-spot; skipped if Gudhi missing)
  • Fractal encoder on spectral band energies (global slide prior, broadcast per-spot)
  • Slepian localized spectral energies (per-spot; k-means region masks)

Variants preserved:
  - baseline_linear, baseline_mlp, abl_spectral_pool (pure morphology spectrum), sno4_reg
"""

import os, json, math, argparse, logging
import numpy as np
import pandas as pd
from PIL import Image

import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.sparse import coo_matrix, csgraph, csr_matrix
from scipy.spatial import cKDTree
import cv2
import torch
import torch.nn as nn

# Optional imports for persistent topology
try:
    import gudhi
    _HAVE_GUDHI = True
except Exception:
    _HAVE_GUDHI = False

log = logging.getLogger("st6")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --------------------- IO ---------------------
def _one(glob_list):
    """Return the first existing path matching any of the patterns in order."""
    import glob, os
    for pat in glob_list:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def load_10x_visium(dataset_dir: str, sample_id: str | None = None, use_image: str = "hires"):
    """
    Robust loader for your GSE210616 layout.

    Accepts:
      - dataset_dir = parent folder containing many samples (GSM*/), plus sample_id="GSM6433602_118C"
      - OR dataset_dir = direct path to a single sample folder (containing spatial/ and filtered_feature_bc_matrix/)

    Returns:
      adata (AnnData), image_rgb (H,W,3 uint8), coords_xy (N,2 float32) where coords are pixel (x,y).
    """
    import os, json, pandas as pd, numpy as np, scanpy as sc
    from PIL import Image
    from scipy import sparse

    # Resolve sample_dir
    if sample_id:
        sample_dir = os.path.join(dataset_dir, sample_id)
    else:
        # If caller passed the sample folder directly, use it; else try to auto-pick one GSM*
        if os.path.isdir(os.path.join(dataset_dir, "spatial")) and \
           os.path.isdir(os.path.join(dataset_dir, "filtered_feature_bc_matrix")):
            sample_dir = dataset_dir
        else:
            # auto-pick first GSM* so script still runs if user forgets --sample_id
            gsm_dirs = [d for d in sorted(os.listdir(dataset_dir)) if d.startswith("GSM")]
            if not gsm_dirs:
                raise FileNotFoundError(f"No GSM* sample folders found under {dataset_dir}")
            sample_dir = os.path.join(dataset_dir, gsm_dirs[0])

    sp = os.path.join(sample_dir, "spatial")
    mtx_dir = os.path.join(sample_dir, "filtered_feature_bc_matrix")
    if not os.path.isdir(sp):      raise FileNotFoundError(f"Missing spatial/: {sp}")
    if not os.path.isdir(mtx_dir): raise FileNotFoundError(f"Missing filtered_feature_bc_matrix/: {mtx_dir}")

    # --- counts: prefer H5, fallback to MTX triplet ---
    h5 = _one([os.path.join(mtx_dir, "*filtered_feature_bc_matrix.h5"),
               os.path.join(mtx_dir, "filtered_feature_bc_matrix.h5")])
    if h5 and os.path.exists(h5):
        adata = sc.read_10x_h5(h5)
    else:
        mtx = _one([os.path.join(mtx_dir, "matrix.mtx.gz"), os.path.join(mtx_dir, "matrix.mtx")])
        bar = _one([os.path.join(mtx_dir, "barcodes.tsv.gz"), os.path.join(mtx_dir, "barcodes.tsv")])
        feat = _one([os.path.join(mtx_dir, "features.tsv.gz"),
                     os.path.join(mtx_dir, "genes.tsv.gz"),
                     os.path.join(mtx_dir, "features.tsv")])
        if not (mtx and bar and feat):
            raise FileNotFoundError(f"Could not find H5 or MTX triplet in {mtx_dir}")
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
    adata.var_names_make_unique()

    # --- spatial files (handle prefixed or generic names) ---
    pos_csv = _one([
        os.path.join(sp, "*tissue_positions_list.csv"),
        os.path.join(sp, "tissue_positions_list.csv"),
        os.path.join(sp, "*tissue_positions.csv"),
        os.path.join(sp, "tissue_positions.csv"),
    ])
    if not pos_csv:
        raise FileNotFoundError(f"No tissue_positions*.csv under {sp}")

    scale_json = _one([os.path.join(sp, "*scalefactors_json.json"),
                       os.path.join(sp, "scalefactors_json.json")])
    if not scale_json:
        raise FileNotFoundError(f"No scalefactors_json.json under {sp}")

    img_png = _one([
        os.path.join(sp, "*tissue_hires_image.png"),
        os.path.join(sp, "tissue_hires_image.png"),
    ]) if use_image == "hires" else _one([
        os.path.join(sp, "*tissue_lowres_image.png"),
        os.path.join(sp, "tissue_lowres_image.png"),
    ])
    if not img_png:
        raise FileNotFoundError(f"No {'hires' if use_image=='hires' else 'lowres'} image found under {sp}")

    # --- read positions (auto header vs no header) ---
    with open(pos_csv, "r") as f:
        first = f.readline().lower()
    header = 0 if "barcode" in first else None
    pos = pd.read_csv(pos_csv, header=header)
    if header is None:
        # old format: 6 columns without header
        pos.columns = ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
    else:
        # normalize column names if a header exists
        # (ensure we have pxl_row_in_fullres / pxl_col_in_fullres)
        cols = [c.strip().lower() for c in pos.columns]
        pos.columns = cols
        # handle variants like pixel_row/column
        if "pxl_row_in_fullres" not in pos.columns and "pixel_row_in_fullres" in pos.columns:
            pos.rename(columns={"pixel_row_in_fullres":"pxl_row_in_fullres"}, inplace=True)
        if "pxl_col_in_fullres" not in pos.columns and "pixel_col_in_fullres" in pos.columns:
            pos.rename(columns={"pixel_col_in_fullres":"pxl_col_in_fullres"}, inplace=True)

    # --- open image & scales ---
    img = Image.open(img_png).convert("RGB")
    with open(scale_json, "r") as f:
        scales = json.load(f)

    # --- align by barcode, keep in-tissue ---
    pos = pos.set_index("barcode")
    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise RuntimeError("No overlapping barcodes between expression and spatial CSV.")
    adata = adata[common, :].copy()
    adata.obs = adata.obs.join(pos.loc[adata.obs_names])
    adata.obs["in_tissue"] = adata.obs["in_tissue"].astype(int)
    adata = adata[adata.obs["in_tissue"] == 1, :].copy()

    # --- coordinates to match chosen image ---
    if use_image == "hires":
        x = adata.obs["pxl_col_in_fullres"].astype(float).values
        y = adata.obs["pxl_row_in_fullres"].astype(float).values
    else:
        s = float(scales["tissue_lowres_scalef"])
        x = adata.obs["pxl_col_in_fullres"].astype(float).values * s
        y = adata.obs["pxl_row_in_fullres"].astype(float).values * s
    coords_xy = np.c_[x, y].astype(np.float32)

    # ensure dense float32 expression
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata, np.array(img), coords_xy


# --------------------- SNO4-style morphology features ---------------------
def rgb_to_he(rgb):
    I = rgb.astype(np.float64) + 1e-6
    OD = -np.log((I/255.0).clip(1e-6, 1.0))
    H = np.array([0.644, 0.717, 0.267], dtype=np.float64)
    E = np.array([0.093, 0.954, 0.283], dtype=np.float64)
    D = np.cross(H, E)
    M = np.stack([H/np.linalg.norm(H), E/np.linalg.norm(E), D/np.linalg.norm(D)], axis=1)
    C = OD.reshape(-1,3) @ np.linalg.inv(M)
    return C[:,0].reshape(rgb.shape[:2]), C[:,1].reshape(rgb.shape[:2])

def local_entropy(gray, bins=32):
    vals = gray.reshape(-1)
    q = np.clip((vals.astype(np.int32)*bins)//256, 0, bins-1)
    hist = np.bincount(q, minlength=bins).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    return float((-(p[p>0]*np.log(p[p>0])).sum())/np.log(bins))

def spot_patch_features(img_rgb, centers_xy, patch_px=64):
    """Original 8-feature SNO4 recipe at each node/spot."""
    h, w, _ = img_rgb.shape
    half = patch_px//2
    feats=[]
    for (x, y) in centers_xy.astype(int):
        x0, x1 = max(0, x-half), min(w, x+half)
        y0, y1 = max(0, y-half), min(h, y+half)
        if x1-x0<8 or y1-y0<8:
            feats.append(np.zeros(8, dtype=np.float32)); continue
        patch = img_rgb[y0:y1, x0:x1, :]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        yy, xx = np.indices(gray.shape)
        cy, cx = (gray.shape[0]-1)/2.0, (gray.shape[1]-1)/2.0
        dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        dist = dist/(dist.max()+1e-12)

        sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sob_mag = np.sqrt(sobx*sobx + soby*soby)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

        Hc, Ec = rgb_to_he(patch)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        S = hsv[...,1]

        f = np.array([
            dist.mean(),
            gray.mean(),
            sob_mag.mean(),
            float(lap.var()),
            local_entropy(gray, bins=32),
            float(Hc.mean()),
            float(Ec.mean()),
            float(S.mean())
        ], dtype=np.float32)
        feats.append(f)
    return np.vstack(feats).astype(np.float32)

# --------------------- Graph (epsilon) + Laplacian ---------------------
def epsilon_graph(coords_xy, eps=None, max_nn=12, sigma=50.0):
    """
    coords_xy in pixels (N,2). If eps None -> 1.5 * median 1-NN distance.
    Weighted symmetric graph: w_ij = exp(-d^2 / sigma^2) for d<=eps.
    """
    N = coords_xy.shape[0]
    tree = cKDTree(coords_xy)
    dists, idxs = tree.query(coords_xy, k=min(max_nn, N))
    if eps is None:
        nn1 = dists[:,1] if dists.shape[1]>1 else dists[:,0]
        eps = 1.5 * np.median(nn1[np.isfinite(nn1)])
        eps = max(eps, 10.0)
    rows, cols, vals = [], [], []
    for i in range(N):
        for j, d in zip(idxs[i], dists[i]):
            if j==i or not np.isfinite(d): continue
            if d<=eps:
                w = math.exp(-(d*d)/(sigma*sigma))
                rows.append(i); cols.append(j); vals.append(w)
    rows2 = rows + cols; cols2 = cols + rows; vals2 = vals + vals
    W = coo_matrix((vals2, (rows2, cols2)), shape=(N, N)).tocsr()
    L = csgraph.laplacian(W, normed=True)
    return W, L

# --------------------- HKS / GWE / Persistent Topology ---------------------
def hks_features(U: np.ndarray, lam: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-spot Heat Kernel Signature: HKS_i(t) = sum_k exp(-t*lam_k) * u_{ik}^2"""
    N, K = U.shape
    T = len(times)
    U2 = (U*U).astype(np.float64)  # (N,K)
    H = np.empty((N, T), dtype=np.float64)
    lam64 = lam.astype(np.float64)
    for j, t in enumerate(times):
        wt = np.exp(-t * lam64)  # (K,)
        H[:, j] = U2 @ wt  # (N,)
    return H.astype(np.float32)

def _gfilter_weights(lam: np.ndarray, scale: float, kind: str) -> np.ndarray:
    lam = lam.astype(np.float64)
    if kind == "heat":
        return np.exp(-scale * lam)
    elif kind == "mexican":
        x = scale * lam
        return x * np.exp(-x)
    else:
        raise ValueError(f"Unknown wavelet kind: {kind}")

def graph_wave_energy_features(X_feat: np.ndarray, U: np.ndarray, lam: np.ndarray,
                               scales: np.ndarray, kind: str = "heat",
                               per_channel: bool = True) -> np.ndarray:
    """Per-spot Graph Wave Energy features.
    Steps: project features to spectrum; apply spectral filter g_s(λ); back-project; take squared magnitude.
    Returns: float32 matrix of shape (N, C*S) if per_channel else (N,S)."""
    N, C = X_feat.shape
    mu = X_feat.mean(axis=0, keepdims=True)
    sd = X_feat.std(axis=0, keepdims=True) + 1e-12
    Xn = (X_feat - mu) / sd

    A = (U.T @ Xn).astype(np.float64)  # (K,C)
    outs = []
    for s in scales:
        g = _gfilter_weights(lam, s, kind).reshape(-1, 1)  # (K,1)
        F = U @ (g * A)                                     # (N,C)
        E = (F * F)                                         # squared magnitude
        outs.append(E.astype(np.float32) if per_channel else E.sum(axis=1, keepdims=True).astype(np.float32))
    return np.concatenate(outs, axis=1).astype(np.float32)

def persistent_patch_features(img_rgb: np.ndarray, centers_xy: np.ndarray,
                              patch_px: int = 64, down_px: int = 32,
                              max_bars: int = 4096) -> np.ndarray:
    """Per-spot persistent topology features from image patches (Gudhi CubicalComplex)."""
    if not _HAVE_GUDHI:
        return np.zeros((centers_xy.shape[0], 8), dtype=np.float32)

    h, w, _ = img_rgb.shape
    half = patch_px // 2
    feats = []
    for (x, y) in centers_xy.astype(int):
        x0, x1 = max(0, x-half), min(w, x+half)
        y0, y1 = max(0, y-half), min(h, y+half)
        if x1-x0 < 8 or y1-y0 < 8:
            feats.append(np.zeros(8, dtype=np.float32)); continue
        patch = img_rgb[y0:y1, x0:x1, :]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (down_px, down_px), interpolation=cv2.INTER_AREA)
        vals = gray.reshape(-1).astype(np.float64)
        cc = gudhi.CubicalComplex(dimensions=[down_px, down_px], top_dimensional_cells=vals)
        cc.persistence(homology_coeff_field=2, min_persistence=0.0)
        d0 = cc.persistence_intervals_in_dimension(0)
        d1 = cc.persistence_intervals_in_dimension(1)
        if len(d0) > max_bars: d0 = d0[:max_bars]
        if len(d1) > max_bars: d1 = d1[:max_bars]
        def summarize(diag):
            if len(diag) == 0:
                return 0, 0.0, 0.0, 0.0
            births = np.array([b for b, _ in diag], dtype=np.float64)
            pers = np.array([(d - b) if np.isfinite(d) else 0.0 for b, d in diag], dtype=np.float64)
            return len(diag), float(pers.sum()), float(pers.max(initial=0.0)), float(births.mean())
        n0, s0, m0, mb0 = summarize(d0)
        n1, s1, m1, mb1 = summarize(d1)
        feats.append(np.array([n0, n1, s0, s1, m0, m1, mb0, mb1], dtype=np.float32))
    return np.vstack(feats).astype(np.float32)

# --------------------- Spectral helpers ---------------------
def _lambda_bands(lam, n_bands):
    lam = np.asarray(lam, float); K=lam.size
    if n_bands<=1 or K<=n_bands:
        edges = np.linspace(0, K, n_bands+1, dtype=int)
    else:
        qs = np.linspace(0,1,n_bands+1)
        cuts = np.unique(np.searchsorted(lam, np.quantile(lam, qs, method="linear")))
        cuts[0]=0
        if cuts[-1]!=K: cuts=np.r_[cuts,K]
        if cuts.size<2: edges = np.linspace(0,K,n_bands+1,dtype=int)
        else: edges = cuts
    bands = [np.arange(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    return [b for b in bands if b.size>0]

# --------------------- Fractal encoder on spectral band energies ---------------------
def _fit_loglog_slope(y, eps=1e-12):
    """Slope & R² of log(y) ~ a*log(j)+b over j=1..K; y>=0."""
    y = np.asarray(y, float)
    K = int(y.size)
    if K < 6 or not np.isfinite(y).all():
        return 0.0, 0.0
    j = np.arange(1, K+1, dtype=np.float64)
    x = np.log(j + eps); t = np.log(y + eps)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    t_hat = a*x + b
    ss_res = float(np.sum((t - t_hat)**2))
    ss_tot = float(np.sum((t - t.mean())**2)) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return float(a), float(max(min(r2, 1.0), 0.0))

def _renyi_dimensions(p, qs=(0.5, 1.0, 2.0, 4.0), eps=1e-12):
    p = np.asarray(p, float); p = p / (p.sum() + eps)
    B = int(p.size); out = []
    logB = np.log(B + eps)
    H = -np.sum(p * np.log(p + eps))  # Shannon
    for q in qs:
        if abs(q - 1.0) < 1e-12:
            out.append(float(H / (logB + eps)))
        else:
            z = float(np.sum(np.power(p + eps, q)))
            out.append(float(np.log(z + eps) / ((q - 1.0) * (logB + eps))))
    return out

def _lacunarity_1d(y, windows=(2, 4, 8), eps=1e-12):
    y = np.asarray(y, float)
    y = y - np.min(y) + eps
    K = int(y.size); vals=[]
    for w in windows:
        if w > K:
            vals.append(1.0); continue
        cums = np.cumsum(np.r_[0.0, y])
        box = cums[w:] - cums[:-w]
        mu = float(np.mean(box)); var = float(np.var(box))
        Λ = var / (mu*mu + eps) + 1.0
        vals.append(float(Λ))
    return vals

def fractal_encode_channels(E, eps=1e-12):
    """E: (C,K) band energies per channel across K bands (L2-normalized along bands).
    Output per-channel: [slope, R2, D0.5, D1, D2, D4, Lac2, Lac4, Lac8] → 9*C dims
    """
    E = np.asarray(E, float); C, K = E.shape
    colsum = np.sum(E, axis=0)
    nz = np.nonzero(colsum > eps)[0]
    if nz.size >= 6: E = E[:, :int(nz.max())+1]
    feats=[]
    for c in range(C):
        y = E[c]
        if y.size < 6 or (not np.isfinite(y).all()) or np.all(y <= eps):
            feats += [0.0, 0.0] + [0.0, 0.0, 0.0, 0.0] + [1.0, 1.0, 1.0]
            continue
        slope, r2 = _fit_loglog_slope(y, eps=eps)
        Dq = _renyi_dimensions(y/(y.sum()+eps), qs=(0.5,1.0,2.0,4.0), eps=eps)
        lac = _lacunarity_1d(y, windows=(2,4,8), eps=eps)
        feats += [slope, r2] + Dq + lac
    return np.array(feats, dtype=np.float32)

# --------------------- Spatial autocorrelation (Moran's I & Geary's C) ---------------------
def _knn_weights(coords_xy: np.ndarray, k: int = 8, row_norm: bool = True):
    """Classic symmetric KNN graph with 1/(1+d) weights; optional row-normalization."""
    N = coords_xy.shape[0]
    k = int(max(1, min(k, N - 1)))
    tree = cKDTree(coords_xy)
    dists, idxs = tree.query(coords_xy, k=k + 1)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j, d in zip(idxs[i][1:], dists[i][1:]):  # skip self
            if not np.isfinite(d):
                continue
            w = 1.0 / (1.0 + d)  # distance-decay
            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)  # symmetrize

    W = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    if row_norm:
        rs = np.array(W.sum(axis=1)).ravel() + 1e-12
        W = W.multiply(1.0 / rs[:, None])

    # Enforce symmetry (safe for Moran/Geary)
    W = (W + W.T) * 0.5
    W = csr_matrix(W)
    S0 = float(W.sum())
    return W, S0

def _gauss_eps_weights(coords_xy: np.ndarray, sigma: float = 50.0, eps: float | None = None, max_nn: int = 24):
    """Symmetric Gaussian ε-graph for spatial AC (no row-normalization): w_ij = exp(-d^2 / sigma^2) if d<=eps."""
    N = coords_xy.shape[0]
    tree = cKDTree(coords_xy)
    k = min(max_nn, N)
    dists, idxs = tree.query(coords_xy, k=k)

    if eps is None or eps <= 0:
        nn1 = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
        eps = 1.5 * float(np.median(nn1[np.isfinite(nn1)]))
        eps = max(eps, 10.0)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j, d in zip(idxs[i], dists[i]):
            if j == i or not np.isfinite(d) or d > eps:
                continue
            w = math.exp(-(d * d) / (sigma * sigma))
            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)

    W = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    W = (W + W.T) * 0.5
    W = csr_matrix(W)
    S0 = float(W.sum())
    return W, S0

def _moran_geary(x: np.ndarray, W, S0: float):
    """Moran's I and Geary's C computed robustly on CSR graph."""
    if not sparse.isspmatrix_csr(W):
        W = W.tocsr()

    x = np.asarray(x, float)
    N = x.size
    mu = float(x.mean())
    v = x - mu
    denom = float((v * v).sum()) + 1e-12

    # Moran's I
    Wx = W @ v
    num_I = float(v @ Wx)
    I = (N / (S0 + 1e-12)) * (num_I / denom)

    # Geary's C (edge-wise)
    diff2_sum = 0.0
    indptr, indices, data = W.indptr, W.indices, W.data
    for i in range(N):
        start, end = indptr[i], indptr[i + 1]
        js = indices[start:end]
        ws = data[start:end]
        if js.size:
            dv = (x[i] - x[js])
            diff2_sum += float((dv * dv * ws).sum())

    C = ((N - 1.0) / (2.0 * (S0 + 1e-12))) * (diff2_sum / denom)
    return float(I), float(C)

def spatial_ac_global_features(coords_xy: np.ndarray, X_feat: np.ndarray,
                               method: str = "gauss",
                               k: int = 8, row_norm: bool = False,
                               sigma: float = 50.0, eps: float | None = None, max_nn: int = 24,
                               zscore: bool = True):
    """Compute global Moran's I and Geary's C per channel of X_feat across the spot graph.
    method: 'gauss' (Gaussian ε-graph, default) or 'knn'."""
    if method == "knn":
        W, S0 = _knn_weights(coords_xy, k=k, row_norm=row_norm)
    else:
        W, S0 = _gauss_eps_weights(coords_xy, sigma=sigma, eps=eps, max_nn=max_nn)
    try:
        log
    except NameError:
        import logging
        log = logging.getLogger("st6")

    log.info(f"[AC] nnz={W.nnz}, S0={S0:.3e}, minW={(W.data.min() if W.nnz else 0):.3e}, maxW={(W.data.max() if W.nnz else 0):.3e}")

    C = X_feat.shape[1]
    out = []
    for j in range(C):
        x = X_feat[:, j].astype(np.float64)
        if zscore:
            x = (x - x.mean()) / (x.std() + 1e-12)
        I, Cg = _moran_geary(x, W, S0)
        out += [I, Cg]
    return np.array(out, dtype=np.float32)

def spatial_ac_global_features_edges(coords_xy: np.ndarray, X_feat: np.ndarray,
                                     method: str = "knn",
                                     k: int = 24, sigma: float = 50.0,
                                     eps: float | None = None, zscore: bool = True):
    """
    Edge-list implementation of Moran's I and Geary's C.
    Builds undirected edges (i < j) with weights w_ij, then uses:
      S0 = 2 * sum(w_ij)
      I  = (N / S0) * ( 2*sum w_ij v_i v_j / sum v_i^2 )
      C  = ((N-1)/(2*S0)) * ( 2*sum w_ij (x_i-x_j)^2 / sum v_i^2 )
    Returns: AC vector [I1, C1, I2, C2, ...], S0, num_edges
    """
    from scipy.spatial import cKDTree

    coords_xy = np.asarray(coords_xy, float)
    N = coords_xy.shape[0]
    C = X_feat.shape[1]
    tree = cKDTree(coords_xy)

    # --- build undirected edges (i<j) + weights ---
    if method == "knn":
        kq = int(min(k + 1, max(2, N)))
        dists, idxs = tree.query(coords_xy, k=kq)
        i_list, j_list, w_list = [], [], []
        for i in range(N):
            js = idxs[i][1:]   # skip self
            ds = dists[i][1:]
            for j, d in zip(js, ds):
                if j <= i or not np.isfinite(d):  # undirected, keep one
                    continue
                w = 1.0 / (1.0 + float(d))
                i_list.append(i); j_list.append(j); w_list.append(w)
    else:
        # Gaussian ε-graph via radius query
        if eps is None or eps <= 0:
            d1 = tree.query(coords_xy, k=2)[0][:, 1]
            eps = max(1.5 * float(np.median(d1[np.isfinite(d1)])), 10.0)
        pairs = list(tree.query_pairs(r=float(eps)))
        if len(pairs) == 0:
            i_list = []; j_list = []; w_list = []
        else:
            i_arr = np.array([i for (i, _) in pairs], dtype=np.int32)
            j_arr = np.array([j for (_, j) in pairs], dtype=np.int32)
            diff = coords_xy[i_arr] - coords_xy[j_arr]
            d = np.sqrt(np.sum(diff * diff, axis=1))
            w = np.exp(-(d * d) / (sigma * sigma))
            i_list = i_arr.tolist(); j_list = j_arr.tolist(); w_list = w.tolist()

    i_arr = np.array(i_list, dtype=np.int32)
    j_arr = np.array(j_list, dtype=np.int32)
    w = np.array(w_list, dtype=np.float64)
    S0 = 2.0 * float(w.sum())
    M = int(i_arr.size)

    # --- per-channel AC ---
    out = []
    for cj in range(C):
        x = X_feat[:, cj].astype(np.float64)
        if zscore:
            x = (x - x.mean()) / (x.std() + 1e-12)
        v = x - x.mean()
        denom = float(np.sum(v * v)) + 1e-12

        if M > 0:
            num_I = 2.0 * float(np.sum(w * (v[i_arr] * v[j_arr])))
            num_C = 2.0 * float(np.sum(w * ((x[i_arr] - x[j_arr]) ** 2)))
        else:
            num_I = 0.0; num_C = 0.0

        I = (N / (S0 + 1e-12)) * (num_I / denom)
        Cg = ((N - 1.0) / (2.0 * (S0 + 1e-12))) * (num_C / denom)
        out += [I, Cg]

    return np.array(out, dtype=np.float32), S0, M

# --------------------- Graph Slepians ---------------------
def slepian_basis(U: np.ndarray, masks: list, Kb: int, m_keep: int):
    """Compute Slepian basis vectors per mask using bandlimited eigenvectors U_Kb.
    Returns list of arrays [G_r] where G_r is (N, m_keep) Slepian vectors (top by concentration)."""
    N, K = U.shape
    Kb = int(min(Kb, K))
    m_keep = int(m_keep)
    U_K = U[:, :Kb].astype(np.float64)  # (N,Kb)
    out = []
    for M_diag in masks:
        # C = U_K^T M U_K, where M is diagonal selection (provided as its diag)
        C = (U_K.T * M_diag.reshape(1, -1)) @ U_K  # (Kb,Kb)
        try:
            vals, vecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            vals = np.ones(Kb, dtype=np.float64)
            vecs = np.eye(Kb, dtype=np.float64)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx[:m_keep]]               # (Kb, m_keep)
        G = (U_K @ vecs).astype(np.float32)        # (N, m_keep)
        out.append(G)
    return out


def slepian_local_energy_features(X_feat: np.ndarray, U: np.ndarray, coords: np.ndarray,
                                  regions: int = 4, m_keep: int = 6, Kb: int = 32,
                                  mode: str = "kmeans") -> np.ndarray:
    """Per-spot Slepian-localized energy features."""
    N, C = X_feat.shape
    # Normalize channels across spots
    mu = X_feat.mean(axis=0, keepdims=True)
    sd = X_feat.std(axis=0, keepdims=True) + 1e-12
    Xn = (X_feat - mu) / sd  # (N,C)

    # Region masks
    if mode == "kmeans":
        R = max(2, int(regions))
        labels = KMeans(n_clusters=R, random_state=42, n_init=10).fit_predict(coords)
        masks = []
        for r in range(R):
            mask_r = (labels == r).astype(np.float64)  # 1 inside region r, else 0
            masks.append(mask_r)
    else:
        R = 1
        masks = [np.ones(N, dtype=np.float64)]

    # Build Slepian bases
    G_list = slepian_basis(U, masks, Kb=int(Kb), m_keep=int(m_keep))  # list of (N, m_keep)

    # Localized reconstruction and energy per spot/channel
    feats = []
    for r, G in enumerate(G_list):
        B = (G.T @ Xn).astype(np.float32)     # (m_keep, C)
        Z = (G @ B).astype(np.float32)        # (N, C)
        E = (Z * Z)                           # energy per spot/channel
        feats.append(E)
    X_slep = np.concatenate(feats, axis=1).astype(np.float32)  # (N, C*R)
    return X_slep


# --------------------- Models ---------------------
class SpectralMix(nn.Module):
    """
    Project X to spectral (U^T X) -> (K,C)
    Band-mix along K, then channel-mix along C, back-project to node space.
    Note: expects X to be (B,C) and U to be (B,K) for a batch of B nodes.
    """
    def __init__(self, C, K, rank=32, nonlin='softplus', pdrop=0.1):
        super().__init__()
        r = min(rank, min(C, K))
        self.band = nn.Sequential(nn.Linear(K, r, bias=False), nn.Linear(r, K, bias=False))
        self.chan = nn.Sequential(nn.Linear(C, r, bias=False), nn.Linear(r, C, bias=False))
        self.act = nn.Softplus() if nonlin == 'softplus' else nn.GELU()
        self.drop = nn.Dropout(pdrop)

    def forward(self, X, U):  # X: (B,C), U: (B,K)
        Fk = U.transpose(0,1) @ X   # (K,C)
        Fk = Fk.transpose(0, 1)     # (C,K)
        Fk = self.band(Fk).transpose(0, 1)  # (K,C)
        Fk = self.chan(Fk)          # (K,C)
        Xp = U @ Fk                 # (B,C)
        return self.drop(self.act(Xp))

class SNO4Regressor(nn.Module):
    def __init__(self, C, K, hidden=64, rank=32, pdrop=0.1, out_dim=1):
        super().__init__()
        self.mix1 = SpectralMix(C, K, rank=rank, pdrop=pdrop)
        self.mix2 = SpectralMix(C, K, rank=rank, pdrop=pdrop)
        self.head = nn.Sequential(
            nn.Linear(C, hidden),
            nn.Softplus(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, X, U):
        z = self.mix1(X, U)
        z = self.mix2(z, U)
        return self.head(z)    # (B, out_dim)

# --------------------- Metrics ---------------------
def pearson_corr(a, b, eps=1e-8):
    a = a - a.mean(); b = b - b.mean()
    num = (a*b).sum()
    den = np.sqrt((a*a).sum())*np.sqrt((b*b).sum()) + eps
    return float(num/den)

# --------------------- Helpers ---------------------
def to_log1p(Y):
    return np.log1p(np.maximum(Y, 0.0).astype(np.float32))

def from_expm1(Y_log):
    return np.expm1(Y_log.astype(np.float32))

def save_metrics(out_dir, variant, genes, folds_mses_log, folds_pcc_log, preds_log, Y_log, region_k, coords):
    # Aggregate per-gene in log-space
    mses_mat = np.vstack(folds_mses_log)
    pccs_mat = np.vstack(folds_pcc_log)
    gene_metrics = pd.DataFrame({
        "gene": genes,
        "MSE_log_mean": mses_mat.mean(axis=0),
        "MSE_log_std":  mses_mat.std(axis=0),
        "PCC_log_mean": pccs_mat.mean(axis=0),
        "PCC_log_std":  pccs_mat.std(axis=0),
    }).sort_values("PCC_log_mean", ascending=False)
    gene_metrics.to_csv(os.path.join(out_dir, f"metrics_{variant}.csv"), index=False)

    # Convert to original space for reporting (MSE_orig, PCC_orig)
    Y_hat = from_expm1(preds_log)
    Y_true = from_expm1(Y_log)
    G = Y_true.shape[1]
    mses_orig = [mean_squared_error(Y_true[:,j], Y_hat[:,j]) for j in range(G)]
    pccs_orig = [pearson_corr(Y_true[:,j], Y_hat[:,j]) for j in range(G)]
    gene_metrics2 = gene_metrics.copy()
    gene_metrics2["MSE_orig_mean"] = mses_orig
    gene_metrics2["PCC_orig_mean"] = pccs_orig
    gene_metrics2.to_csv(os.path.join(out_dir, f"metrics_with_orig_{variant}.csv"), index=False)

    # Save predictions (log-space)
    np.savetxt(os.path.join(out_dir, f"predictions_{variant}.csv"),
               preds_log, delimiter=",", fmt="%.6f")

    # Region-level PCC (cluster spots)
    k = max(2, min(region_k, coords.shape[0]))
    regs = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
    rows=[]
    for j,g in enumerate(genes):
        y = Y_true[:,j]; yhat = Y_hat[:,j]
        df = pd.DataFrame({"reg": regs, "y": y, "yhat": yhat}).groupby("reg").mean()
        pcc = pearson_corr(df["y"].values, df["yhat"].values) if df.shape[0]>1 else np.nan
        rows.append({"gene": g, "RegionPCC_orig": pcc})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"region_pcc_{variant}.csv"), index=False)

# --------------------- Variants (Baselines & Ablations) ---------------------
def run_variant_baseline_linear(X_feat, Y_log, folds, out_dir, variant, genes, coords, regions):
    from sklearn.linear_model import Ridge
    preds = np.zeros_like(Y_log, dtype=np.float32)
    folds_mses, folds_pcc = [], []
    for fold, (tr, va) in enumerate(folds, 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_feat[tr])
        Xva = scaler.transform(X_feat[va])
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(Xtr, Y_log[tr])
        yhat = model.predict(Xva).astype(np.float32)
        preds[va] = yhat
        G = Y_log.shape[1]
        mses = [mean_squared_error(Y_log[va, j], yhat[:, j]) for j in range(G)]
        pccs = [pearson_corr(Y_log[va, j], yhat[:, j]) for j in range(G)]
        log.info(f"[{variant}] Fold {fold}: MSE_LOG mean={np.mean(mses):.4f} | PCC_LOG mean={np.mean(pccs):.4f}")
        folds_mses.append(mses); folds_pcc.append(pccs)
    save_metrics(out_dir, variant, genes, folds_mses, folds_pcc, preds, Y_log, regions, coords)

def run_variant_baseline_mlp(X_feat, Y_log, folds, out_dir, variant, genes, coords, regions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class MLP(nn.Module):
        def __init__(self, C, G):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(C, 128), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64), nn.Softplus(),
                nn.Linear(64, G)
            )
        def forward(self, x): return self.net(x)

    preds = np.zeros_like(Y_log, dtype=np.float32)
    folds_mses, folds_pcc = [], []
    for fold, (tr, va) in enumerate(folds, 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_feat[tr]).astype(np.float32)
        Xva = scaler.transform(X_feat[va]).astype(np.float32)
        ytr = Y_log[tr].astype(np.float32); yva = Y_log[va].astype(np.float32)

        model = MLP(Xtr.shape[1], Y_log.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)

        model.train()
        bs = 2048
        idx = np.arange(Xtr.shape[0])
        for _ in range(60):
            np.random.shuffle(idx)
            for s in range(0, len(idx), bs):
                ii = idx[s:s+bs]
                xb = Xtr_t[ii]; yb = ytr_t[ii]
                pred = model(xb); loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            yhat = model(Xva_t).cpu().numpy().astype(np.float32)
        preds[va] = yhat

        G = Y_log.shape[1]
        mses = [mean_squared_error(yva[:, j], yhat[:, j]) for j in range(G)]
        pccs = [pearson_corr(yva[:, j], yhat[:, j]) for j in range(G)]
        log.info(f"[{variant}] Fold {fold}: MSE_LOG mean={np.mean(mses):.4f} | PCC_LOG mean={np.mean(pccs):.4f}")
        folds_mses.append(mses); folds_pcc.append(pccs)
    save_metrics(out_dir, variant, genes, folds_mses, folds_pcc, preds, Y_log, regions, coords)

def run_variant_spectral_pool(X_feat, U, lam, Y_log, folds, out_dir, variant, genes, coords, regions, n_bands=14, mode="invariant"):
    """
    Per-spot spectral features via band filtering (ABLATION, unchanged)
    """
    N, C = X_feat.shape
    mu = X_feat.mean(axis=0, keepdims=True)
    sd = X_feat.std(axis=0, keepdims=True) + 1e-12
    Xn = (X_feat - mu) / sd

    a = (U.T @ Xn).astype(np.float32)
    bands = _lambda_bands(lam, n_bands)

    if mode == "aligned":
        # simple anchors: coord-centered + first channel
        g0 = X_feat[:,0].astype(np.float64)
        g0 = (g0 - g0.mean())/(np.linalg.norm(g0)+1e-12)
        y = (coords[:,1] - coords[:,1].mean()); y = y/(np.linalg.norm(y)+1e-12)
        x = (coords[:,0] - coords[:,0].mean()); x = x/(np.linalg.norm(x)+1e-12)
        anchors = np.stack([g0,y,x], axis=1)
        U_rot = U.copy(); a_rot = a.copy()
        for idx in bands:
            UB = U_rot[:, idx]; S = UB.T @ anchors
            try:
                R,_,_ = np.linalg.svd(S, full_matrices=False)
            except np.linalg.LinAlgError:
                R = np.eye(idx.size, dtype=np.float32)
            U_rot[:, idx] = UB @ R
            a_rot[idx, :] = (R.T @ a_rot[idx, :])
        U_use, a_use = U_rot, a_rot
    else:
        U_use, a_use = U, a

    Z_blocks = []
    for idx in bands:
        Zb = (U_use[:, idx] @ a_use[idx, :]).astype(np.float32)  # (N,C)
        Z_blocks.append(Zb)
    Z_cat = np.concatenate(Z_blocks, axis=1)  # (N, C*B)

    B = len(bands)
    Z_nbC = np.stack([Zb for Zb in Z_blocks], axis=1)  # (N, B, C)
    ch_mean = Z_nbC.mean(axis=1)  # (N, C)
    ch_std  = Z_nbC.std(axis=1)   # (N, C)

    X_spec = np.concatenate([Z_cat, ch_mean, ch_std], axis=1).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class MLP(nn.Module):
        def __init__(self, D, G):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D, 256), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128), nn.Softplus(),
                nn.Linear(128, G)
            )
        def forward(self, x): return self.net(x)

    preds = np.zeros_like(Y_log, dtype=np.float32)
    folds_mses, folds_pcc = [], []
    for fold, (tr, va) in enumerate(folds, 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_spec[tr]).astype(np.float32)
        Xva = scaler.transform(X_spec[va]).astype(np.float32)
        ytr = Y_log[tr].astype(np.float32); yva = Y_log[va].astype(np.float32)

        model = MLP(Xtr.shape[1], Y_log.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)

        model.train()
        bs = 2048
        idx = np.arange(Xtr.shape[0])
        for _ in range(60):
            np.random.shuffle(idx)
            for s in range(0, len(idx), bs):
                ii = idx[s:s+bs]
                xb = Xtr_t[ii]; yb = ytr_t[ii]
                pred = model(xb); loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            yhat = model(Xva_t).cpu().numpy().astype(np.float32)
        preds[va] = yhat

        G = Y_log.shape[1]
        mses = [mean_squared_error(yva[:, j], yhat[:, j]) for j in range(G)]
        pccs = [pearson_corr(yva[:, j], yhat[:, j]) for j in range(G)]
        log.info(f"[{variant}] Fold {fold}: MSE_LOG mean={np.mean(mses):.4f} | PCC_LOG mean={np.mean(pccs):.4f}")
        folds_mses.append(mses); folds_pcc.append(pccs)
    save_metrics(out_dir, variant, genes, folds_mses, folds_pcc, preds, Y_log, regions, coords)

def run_variant_sno4(X_feat, U, Y_log, folds, out_dir, variant, genes, coords, regions,
                     hidden=64, rank=32, dropout=0.10, epochs=80, batch=2048, wd=1e-4, lr=3e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C = X_feat.shape; G = Y_log.shape[1]; K = U.shape[1]
    preds = np.zeros_like(Y_log, dtype=np.float32)
    folds_mses, folds_pcc = [], []
    for fold, (tr, va) in enumerate(folds, 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_feat[tr]).astype(np.float32)
        Xva = scaler.transform(X_feat[va]).astype(np.float32)
        ytr = Y_log[tr].astype(np.float32); yva = Y_log[va].astype(np.float32)
        Utr = U[tr].astype(np.float32)
        Uva = U[va].astype(np.float32)

        model = SNO4Regressor(C=C, K=K, hidden=hidden, rank=rank, pdrop=dropout, out_dim=G).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)
        Utr_t = torch.from_numpy(Utr).to(device)
        Uva_t = torch.from_numpy(Uva).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)

        model.train()
        rng = np.random.default_rng(42+fold)
        idx = np.arange(Xtr.shape[0])
        for _ in range(epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), batch):
                ii = idx[s:s+batch]
                xb = Xtr_t[ii]
                ub = Utr_t[ii]
                yb = ytr_t[ii]
                pred = model(xb, ub)
                opt.zero_grad(); loss = loss_fn(pred, yb); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            yhat = model(Xva_t, Uva_t).cpu().numpy().astype(np.float32)
        preds[va] = yhat

        mses = [mean_squared_error(yva[:, j], yhat[:, j]) for j in range(G)]
        pccs = [pearson_corr(yva[:, j], yhat[:, j]) for j in range(G)]
        log.info(f"[{variant}] Fold {fold}: MSE_LOG mean={np.mean(mses):.4f} | PCC_LOG mean={np.mean(pccs):.4f}")
        folds_mses.append(mses); folds_pcc.append(pccs)
    save_metrics(out_dir, variant, genes, folds_mses, folds_pcc, preds, Y_log, regions, coords)

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    # --- paths & core ---
    ap.add_argument("--dataset_dir", default="/fs/scratch/PAS2942/Alejandro/datasets/ST_10x_Breast")
    ap.add_argument("--out_dir", default="/fs/scratch/PAS2942/Alejandro/datasets/ST_10x_Breast_out/SUGOMA2")
    ap.add_argument("--patch_px", type=int, default=64)
    ap.add_argument("--K", type=int, default=64, help="# eigenvectors")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--genes", type=int, default=50, help="# top-HVGs; or pass --gene_list")
    ap.add_argument("--gene_list", type=str, default="")
    ap.add_argument("--regions", type=int, default=8)
    ap.add_argument("--sample_id", type=str, default="", help="If dataset_dir is a parent folder, pick a GSMxxxx_xxx sample")
    ap.add_argument("--use_image", choices=["hires","lowres"], default="hires")

    # --- spectral pool ---
    ap.add_argument("--n_bands", type=int, default=14)
    ap.add_argument("--spectral_mode", choices=["invariant","aligned"], default="invariant")

    # --- feature flags ---
    ap.add_argument("--use_hks", type=int, default=1, help="Add HKS features (1/0)")
    ap.add_argument("--hks_times", type=str, default="0.001,0.01,0.1,1.0", help="Comma list of HKS times")
    ap.add_argument("--use_gwe", type=int, default=1, help="Add Graph Wave Energy features (1/0)")
    ap.add_argument("--gwe_scales", type=str, default="0.25,0.5,1.0,2.0", help="Comma list of wavelet scales")
    ap.add_argument("--gwe_kind", choices=["heat","mexican"], default="heat")
    ap.add_argument("--use_persistent", type=int, default=1, help="Add persistent topology patch features (1/0)")
    ap.add_argument("--pers_down_px", type=int, default=32, help="Downsample patch to this for Gudhi cubical PH")

    # --- fractal flags ---
    ap.add_argument("--use_fractal", type=int, default=1, help="Add fractal encodings on spectral band energies (1/0)")
    ap.add_argument("--fractal_bands", type=int, default=14, help="# bands for fractal encoder (use eigenvalue-quantile bins)")

    # --- spatial autocorr flags (default: Gaussian ε-graph, no row-norm) ---
    ap.add_argument("--use_spatial_ac", type=int, default=1, help="Add Moran's I and Geary's C on base channels (1/0)")
    ap.add_argument("--ac_method", choices=["gauss","knn"], default="gauss", help="Spatial weights type")
    ap.add_argument("--ac_k", type=int, default=24, help="k for KNN spatial weights")
    ap.add_argument("--ac_row_norm", type=int, default=0, help="Row-normalize KNN weights (1/0)")
    ap.add_argument("--ac_sigma", type=float, default=50.0, help="Sigma for Gaussian ε-graph (pixels)")
    ap.add_argument("--ac_eps", type=float, default=0.0, help="Neighborhood radius; <=0 means auto from median 1-NN")
    ap.add_argument("--ac_max_nn", type=int, default=24, help="Max neighbors probed before ε cutoff (gauss)")
    ap.add_argument("--ac_zscore", type=int, default=1, help="Z-score channels before AC (1/0)")

    # --- Slepian flags ---
    ap.add_argument("--use_slepian", type=int, default=1, help="Add Slepian localized energy features (1/0)")
    ap.add_argument("--slepian_regions", type=int, default=4, help="# of k-means regions to define masks")
    ap.add_argument("--slepian_m", type=int, default=6, help="# of Slepian vectors to keep per region")
    ap.add_argument("--slepian_band", type=int, default=32, help="Bandlimit Kb (<=K) for Slepians")
    ap.add_argument("--slepian_mode", choices=["kmeans","tissue"], default="kmeans")

    # --- control which variants get augmented features ---
    ap.add_argument("--augment_all_variants", type=int, default=1,
                    help="If 1, feed augmented X to all variants except abl_spectral_pool (kept pure). If 0, only new *_aug variants use augmented X.")

    # --- variants ---
    ap.add_argument("--variants", type=str,
                    default="baseline_linear,baseline_mlp,abl_spectral_pool,sno4_reg",
                    help="Comma-separated list of variants to run.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log.info("Load Visium…")
    sample_id = args.sample_id if args.sample_id else None
    adata, img, coords = load_10x_visium(args.dataset_dir, sample_id=sample_id, use_image=args.use_image)

    adata.write(os.path.join(args.out_dir, "adata.h5ad"))
    pd.DataFrame(coords, columns=["x_px","y_px"]).to_csv(os.path.join(args.out_dir, "coords_px.csv"), index=False)


    # in-tissue filter
    m_in = (adata.obs["in_tissue"]==1).values
    adata = adata[m_in, :].copy()

    N_spots = coords.shape[0]

    # targets (genes)
    if args.gene_list:
        genes = [g.strip() for g in args.gene_list.split(",") if g.strip()]
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            raise RuntimeError("None of the specified genes found in adata.var_names.")
    else:
        X = adata.X
        if sparse.issparse(X):
            mean = X.mean(axis=0).A1
            X2 = X.copy(); X2.data **= 2
            ex2 = X2.mean(axis=0).A1
            variances = ex2 - mean**2
        else:
            Xdense = np.asarray(X, dtype=np.float32)
            variances = Xdense.var(axis=0)
        idx = np.argsort(variances)[::-1][:args.genes]
        genes = list(adata.var_names[idx])

    subX = adata[:, genes].X
    if sparse.issparse(subX):
        Y = subX.toarray().astype(np.float32)
    else:
        Y = np.asarray(subX, dtype=np.float32)
    Y_log = np.log1p(np.maximum(Y, 0.0)).astype(np.float32)

    # Base morphology encodings
    log.info("Extract morphology (SNO4) features…")
    X_base = spot_patch_features(img, coords, patch_px=args.patch_px)  # (N,8)

    # Graph + spectrum
    log.info("Build epsilon graph + Laplacian spectrum…")
    W_eps, L = epsilon_graph(coords, eps=None, max_nn=12, sigma=50.0)
    K = min(args.K, L.shape[0])
    try:
        from scipy.sparse.linalg import eigsh
        vals, vecs = eigsh(L, k=K, which='SM')
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray()); vecs = vecs[:, :K]
    U = vecs.astype(np.float32)  # (N,K)
    lam = vals.astype(np.float32)

    # --- feature groups ---
    feats_to_concat = [X_base]

    if args.use_hks:
        times = np.array([float(t.strip()) for t in args.hks_times.split(',') if t.strip()], dtype=np.float64)
        log.info(f"Compute HKS with times={times.tolist()} …")
        H = hks_features(U, lam, times)
        feats_to_concat.append(H)
        log.info(f"HKS shape: {H.shape}")

    if args.use_gwe:
        scales = np.array([float(s.strip()) for s in args.gwe_scales.split(',') if s.strip()], dtype=np.float64)
        log.info(f"Compute Graph Wave Energy kind={args.gwe_kind} scales={scales.tolist()} …")
        GWE = graph_wave_energy_features(X_base, U, lam, scales, kind=args.gwe_kind, per_channel=True)
        feats_to_concat.append(GWE)
        log.info(f"GWE shape: {GWE.shape}")

    if args.use_persistent:
        if not _HAVE_GUDHI:
            log.warning("Gudhi not installed — persistent topology patch features SKIPPED.")
        else:
            log.info(f"Compute Persistent Topology on patches (down_px={args.pers_down_px}) …")
            PT = persistent_patch_features(img, coords, patch_px=args.patch_px, down_px=args.pers_down_px)
            feats_to_concat.append(PT)
            log.info(f"Persistent patch topo shape: {PT.shape}")

    # Fractal encoder (GLOBAL → broadcast per-spot)
    if args.use_fractal:
        log.info(f"Compute Fractal encoder on band energies (bands={args.fractal_bands}) …")
        mu = X_base.mean(axis=0, keepdims=True)
        sd = X_base.std(axis=0, keepdims=True) + 1e-12
        Xn = (X_base - mu) / sd                       # (N,C)
        A = (U.T @ Xn).astype(np.float32)             # (K,C)
        bands = _lambda_bands(lam, args.fractal_bands)
        C = X_base.shape[1]; B = len(bands)
        E = np.zeros((C, B), np.float32)
        for j, idx in enumerate(bands):
            E[:, j] = np.sum(A[idx, :]**2, axis=0)    # per-channel energy per band
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)  # L2 normalize per channel across bands
        FR = fractal_encode_channels(E)               # (9*C,)
        FR_tile = np.repeat(FR[None, :], X_base.shape[0], axis=0)
        feats_to_concat.append(FR_tile)
        np.save(os.path.join(args.out_dir, "X_fractal.npy"), FR_tile)
        log.info(f"Fractal features shape: {FR_tile.shape}")

    # Spatial autocorrelation (GLOBAL → broadcast per-spot)
        # Spatial autocorrelation (GLOBAL → broadcast per-spot)
        # Spatial autocorrelation (GLOBAL → broadcast per-spot) — edge-list formulation
    if args.use_spatial_ac:
        # Use all features accumulated so far (before AC) to avoid circularity
        X_sofar = np.concatenate(feats_to_concat, axis=1).astype(np.float32)

        log.info(
            f"Compute Moran's I & Geary's C via {args.ac_method} "
            f"(k={args.ac_k}, row_norm={bool(args.ac_row_norm)}, "
            f"sigma={args.ac_sigma}, eps={args.ac_eps}, "
            f"zscore={bool(args.ac_zscore)}) …"
        )
        AC, S0, M = spatial_ac_global_features_edges(
            coords, X_sofar,
            method=("knn" if args.ac_method == "knn" else "gauss"),
            k=int(args.ac_k),
            sigma=float(args.ac_sigma),
            eps=(None if float(args.ac_eps) <= 0 else float(args.ac_eps)),
            zscore=bool(args.ac_zscore)
        )

        # Diagnostics: show graph & channel variability
        sd = X_sofar.std(axis=0)
        log.info(f"[AC] edges={M}, S0={S0:.3e}, channel_std_range=[{sd.min():.3e}, {sd.max():.3e}]")
        log.info("Spatial autocorrelation global values (first few): " +
                 ", ".join([f"{v:.6f}" for v in AC[:min(10, len(AC))]]))

        # Save & broadcast
        ac_pairs = AC.reshape(-1, 2)
        pd.DataFrame(ac_pairs, columns=["MoranI", "GearyC"]).to_csv(
            os.path.join(args.out_dir, "spatial_ac_global.csv"), index=False
        )
        AC_tile = np.repeat(AC[None, :], X_sofar.shape[0], axis=0)
        feats_to_concat.append(AC_tile)
        np.save(os.path.join(args.out_dir, "X_spatialac.npy"), AC_tile)
        log.info(f"Spatial AC features shape: {AC_tile.shape}")



    # Slepian localized features (per-spot)
    if args.use_slepian:
        log.info(f"Compute Slepian localized energies (regions={args.slepian_regions}, m={args.slepian_m}, band={args.slepian_band}, mode={args.slepian_mode}) …")
        X_slep = slepian_local_energy_features(
            X_base, U, coords,
            regions=int(args.slepian_regions),
            m_keep=int(args.slepian_m),      # <-- renamed
            Kb=int(args.slepian_band),
            mode=args.slepian_mode
            )

        feats_to_concat.append(X_slep)
        np.save(os.path.join(args.out_dir, "X_slepian.npy"), X_slep)
        log.info(f"Slepian features shape: {X_slep.shape}")

    # Concatenate all enabled features
    X_aug = np.concatenate(feats_to_concat, axis=1).astype(np.float32)
    np.save(os.path.join(args.out_dir, "X_base.npy"), X_base)
    np.save(os.path.join(args.out_dir, "X_aug.npy"), X_aug)
    log.info(f"X_base shape={X_base.shape} | X_aug shape={X_aug.shape}")

    # 5-fold CV over spots (single-slide dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(N_spots)
    folds = list(kf.split(indices))

    # Run requested variants
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in variants:
        log.info(f"\n==== Run variant: {variant} ====")
        # Choose which X to feed
        if args.augment_all_variants:
            use_aug = (variant != "abl_spectral_pool")
        else:
            use_aug = variant.endswith("_aug")
        X_for_variant = X_aug if use_aug else X_base

        if variant == "baseline_linear" or variant == "baseline_linear_aug":
            run_variant_baseline_linear(X_for_variant, Y_log, folds, args.out_dir, variant, genes, coords, args.regions)
        elif variant == "baseline_mlp" or variant == "baseline_mlp_aug":
            run_variant_baseline_mlp(X_for_variant, Y_log, folds, args.out_dir, variant, genes, coords, args.regions)
        elif variant == "abl_spectral_pool":
            run_variant_spectral_pool(X_base, U, lam, Y_log, folds, args.out_dir, variant, genes, coords, args.regions,
                                      n_bands=args.n_bands, mode=args.spectral_mode)
        elif variant == "sno4_reg" or variant == "sno4_reg_aug":
            run_variant_sno4(X_for_variant, U, Y_log, folds, args.out_dir, variant, genes, coords, args.regions,
                             hidden=args.hidden, rank=args.rank, dropout=args.dropout,
                             epochs=args.epochs, batch=args.batch, wd=args.wd, lr=args.lr)
        else:
            log.info(f"[skip] Unknown variant: {variant}")

    log.info("Done.")

if __name__ == "__main__":
    main()
