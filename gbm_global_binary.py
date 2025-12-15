#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL GBM region (multi-class) pipeline (five-result suite)

Task: Recreate the breakhis_global_binary pipeline but for the
patch-level region classification task defined by the FULL(HKS GFT CNN GBM SLEP WKS).py script.

We CONSERVE all features, extractions, models, and tests from
breakhis_global_binary:
  1) Baseline CNN→LGBM (image-only)
  2) CNN ⊕ SNO (late fusion with PCA + Hadamard) → LGBM
  3) Spectral pooled (FULL) → LGBM (graph spectral pooled stats + extras)
  4) Spectral pooled (FULL) → LogisticRegression L2
  5) SNO → LGBM (use SNO pooled embedding as features for LGBM)

And we preserve the extended 'FULL' pooled features introduced in
breakhis_global_binary:
  - Fractal descriptors on spectral band energies (slope, Rényi Dq, lacunarity)
  - Heat Kernel Signature (HKS) pooled over nodes
  - Graph spectral wavelet energy (multi-scale heat kernels)
  - Slepian concentration features (energy concentration in ROI)
  - Persistent homology summaries (Gudhi) on region centroids

Change vs. breakhis_global_binary:
  - Dataset/Task: Instead of BreaKHis benign/malignant, we index GBM
    regions by extracting per-contour patches from IMAGE_DIR and
    MASK_DIR (as in the FULL pipeline) and classify the region class.
  - Multi-class: Models, metrics, and AUC handling are generalized to
    multi-class (macro-averaged OvR AUC where applicable).
  - Grouped CV: StratifiedGroupKFold using the source image filename as
    the group (same spirit as FULL's grouped splitting).

CLI example:
python gbm_global_multiclass.py \
  --image_dir /path/to/GBM/Test_Images \
  --mask_dir  /path/to/GBM/Test_Ground_Truth \
  --out_dir   ./out_gbm \
  --build_patches \
  --spectral_mode invariant --bands 14 --k_reg 128

Requires: pygsp, numpy, pandas, scikit-learn, lightgbm (optional), torch,
          torchvision (optional), gudhi (optional), opencv-python, imblearn
"""

import os, re, math, argparse, json, logging, warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gbm_global")
logging.getLogger("pygsp").setLevel(logging.ERROR)

# ---------- optional deps ----------
try:
    import pygsp as pg
    HAS_PYGSP = True
except Exception:
    HAS_PYGSP = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# torchvision is optional; CNN paths skipped if unavailable
try:
    import torchvision.models as tvm
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse.linalg import eigsh

# Optional persistent homology
try:
    import gudhi
    HAS_GUDHI = True
except Exception:
    HAS_GUDHI = False


# ---------- JSON helpers ----------
def _to_jsonable(obj):
    """Recursively convert numpy types/arrays into JSON-serializable Python types."""
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (_np.floating, _np.integer)):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    return obj

# ====================== GBM region labels (color → class) ======================
# Default matches the FULL script; can be overridden via --labels_json
DEFAULT_GBM_LABELS = {
    "Leading_Edge": (33, 143, 166),
    "Infiltrating_Tumor": (210, 5, 208),
    "Cellular_Tumor": (5, 208, 4),
    "Perinecrotic_Zone": (37, 209, 247),
    "Pseudopalisading_Necrosis": (6, 208, 170),
    "Microvascular_Proliferation": (255, 102, 0),
    "Necrosis": (5, 5, 5),
    "Background": (255, 255, 255)
}

# ====================== defaults ======================
DEF_SEED = 42
DEF_FOLDS = 5
DEF_GRID_N = 64
DEF_CELL_THR = 0.20
DEF_W_POS_SIGMA = 10.0
DEF_W_INT_SIGMA = 30.0
DEF_USE_INT_WEIGHT = True
DEF_K_REG = 128

DEF_SNO_EPOCHS = 200
DEF_SNO_BATCH = 64
DEF_SNO_RANK = 32
DEF_SNO_HIDDEN = 64
DEF_SNO_DROPOUT = 0.10
DEF_SNO_LR = 3e-3
DEF_SNO_WD = 1e-4

DEF_HKS_T = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # heat scales
DEF_WAVE_T = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # wavelet (heat) scales

# ====================== utils ======================
def set_global_seed(seed=DEF_SEED):
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def read_rgb(path):
    im = cv2.imread(path)
    if im is None: return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def tissue_mask_largest(rgb, min_area_frac=0.20):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(rgb[bw==255]) > np.mean(rgb[bw==0]): bw = 255 - bw
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    num, labels = cv2.connectedComponents(bw)
    if num <= 1: return None, None
    largest = 1 + np.argmax([(labels==i).sum() for i in range(1, num)])
    mask = (labels==largest).astype(np.uint8)*255
    area_frac = mask.sum() / float(rgb.shape[0]*rgb.shape[1]*255)
    if area_frac < min_area_frac: return None, None
    return mask, gray

# ====================== GBM: instances & patches ======================
def extract_region_mask(mask_rgb, target_rgb, tol=2):
    diff = np.abs(mask_rgb.astype(np.int16) - np.array(target_rgb, dtype=np.int16))
    return ((diff <= tol).all(axis=-1).astype(np.uint8) * 255)

def extract_contours(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def contour_crop(rgb, contour, pad=8):
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask > 0)
    if ys.size == 0: return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); y1 = min(rgb.shape[0]-1, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(rgb.shape[1]-1, x1 + pad)
    patch = rgb[y0:y1+1, x0:x1+1].copy()
    crop_mask = mask[y0:y1+1, x0:x1+1]
    patch[crop_mask == 0] = 255  # white background
    return patch

def build_or_load_gbm_instances(image_dir: str, mask_dir: str, patch_dir: str,
                                labels: Dict[str, Tuple[int,int,int]], tol: int = 2,
                                min_contour_pts: int = 40, force_rebuild: bool = False) -> List[Dict]:
    os.makedirs(patch_dir, exist_ok=True)
    meta_json = Path(patch_dir) / "instances_meta.json"
    if meta_json.exists() and not force_rebuild:
        try:
            with open(meta_json, "r") as f:
                meta = json.load(f)
            log.info(f"[GBM] loaded meta: {meta_json}")
            return meta
        except Exception:
            pass

    imgs = sorted([f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
    msks = sorted([f for f in os.listdir(mask_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
    meta = []
    kernel = np.ones((3,3), np.uint8)
    for img_file, mask_file in tqdm(list(zip(imgs, msks)), total=min(len(imgs), len(msks)), desc="[GBM] build instances"):
        img_path  = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir,  mask_file)
        rgb  = read_rgb(img_path)
        mrgb = read_rgb(mask_path)
        if rgb is None or mrgb is None:
            log.info(f"[GBM] skip missing: {img_file}"); continue
        for region_name, rgb_label in labels.items():
            binmask = extract_region_mask(mrgb, rgb_label, tol=tol)
            if binmask.sum() == 0: continue
            binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, kernel)
            contours = extract_contours(binmask)
            for cidx, contour in enumerate(contours):
                if len(contour) < min_contour_pts: continue
                patch = contour_crop(rgb, contour, pad=8)
                if patch is None or patch.size == 0: continue
                rid = f"{os.path.splitext(img_file)[0]}__{region_name}__{cidx:04d}.png"
                out_path = os.path.join(patch_dir, rid)
                cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                meta.append({
                    "image": img_file,
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "contour_id": cidx,
                    "region": region_name,
                    "patch_path": out_path
                })
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[GBM] instances saved: {meta_json} (N={len(meta)})")
    return meta

# ====================== features & graph (from breakhis_global_binary) ======================

def _rgb_to_he(rgb):
    I = rgb.astype(np.float64) + 1e-6
    OD = -np.log((I/255.0).clip(1e-6, 1.0))
    H = np.array([0.644, 0.717, 0.267], dtype=np.float64)
    E = np.array([0.093, 0.954, 0.283], dtype=np.float64)
    D = np.cross(H, E)
    M = np.stack([H/np.linalg.norm(H), E/np.linalg.norm(E), D/np.linalg.norm(D)], axis=1)
    Minv = np.linalg.inv(M)
    C = OD.reshape(-1,3) @ Minv
    Hc = C[:,0].reshape(rgb.shape[:2])
    Ec = C[:,1].reshape(rgb.shape[:2])
    return Hc, Ec

def _local_entropy(gray, mask, bins=32):
    m = (mask > 0)
    if m.sum() < 8: return 0.0
    vals = gray[m]
    q = np.clip((vals.astype(np.int32) * bins) // 256, 0, bins-1)
    hist = np.bincount(q, minlength=bins).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    return float((-(p[p>0] * np.log(p[p>0])).sum()) / np.log(bins))

def region_graph_pygsp(rgb, gray, mask,
                       grid_n=DEF_GRID_N, cell_thr=DEF_CELL_THR,
                       pos_sigma=DEF_W_POS_SIGMA, int_sigma=DEF_W_INT_SIGMA,
                       use_int_weight=DEF_USE_INT_WEIGHT):
    """Build coarse region graph and 8-channel node features."""
    if not HAS_PYGSP: raise RuntimeError("PyGSP not installed. pip install pygsp")
    h, w = mask.shape
    gh = grid_n; gw = grid_n
    ys = np.linspace(0, h, gh + 1, dtype=int)
    xs = np.linspace(0, w, gw + 1, dtype=int)
    dist = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() > 0: dist = dist / (dist.max() + 1e-12)
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sob_mag = np.sqrt(sobx*sobx + soby*soby)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1].astype(np.float64) / 255.0
    Hc, Ec = _rgb_to_he(rgb)
    nodes=[]
    for iy in range(gh):
        for ix in range(gw):
            y0,y1 = ys[iy], ys[iy+1]
            x0,x1 = xs[ix], xs[ix+1]
            if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                continue
            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            
            frac = (cell > 0).mean() if cell.size else 0.0
            if not np.isfinite(frac) or frac < cell_thr:
                continue
            gyc,gxc = (iy+0.5),(ix+0.5)
            m = (cell>0)
            sub_gray = gray[y0:y1, x0:x1].astype(np.float64)
            sub_dist = dist[y0:y1, x0:x1].astype(np.float64)
            sub_sob  = sob_mag[y0:y1, x0:x1]
            sub_lap  = lap[y0:y1, x0:x1]
            sub_sat  = sat[y0:y1, x0:x1]
            sub_Hc   = Hc[y0:y1, x0:x1]
            sub_Ec   = Ec[y0:y1, x0:x1]
            if m.any():
                mean_int = float(sub_gray[m].mean())
                mean_dst = float(sub_dist[m].mean())
                mean_sob = float(sub_sob[m].mean())
                var_lap  = float(sub_lap[m].var())
                ent_loc  = _local_entropy(sub_gray, cell)
                mean_H   = float(np.maximum(sub_Hc[m], 0).mean())
                mean_E   = float(np.maximum(sub_Ec[m], 0).mean())
                mean_sat = float(sub_sat[m].mean())
            else:
                mean_int = float(np.mean(sub_gray)) if sub_gray.size else 0.0
                mean_dst = float(np.mean(sub_dist)) if sub_dist.size else 0.0
                mean_sob = float(np.mean(sub_sob))  if sub_sob.size  else 0.0
                var_lap  = float(np.var(sub_lap))   if sub_lap.size  else 0.0
                ent_loc  = 0.0
                mean_H   = float(np.mean(np.maximum(sub_Hc, 0))) if sub_Hc.size else 0.0
                mean_E   = float(np.mean(np.maximum(sub_Ec, 0))) if sub_Ec.size else 0.0
                mean_sat = float(np.mean(sub_sat)) if sub_sat.size else 0.0
            nodes.append((iy,ix,gyc,gxc, mean_dst,mean_int,mean_sob,var_lap, ent_loc,mean_H,mean_E,mean_sat))
    if not nodes: return None, None, None
    idx_map = -np.ones((gh,gw), dtype=int)
    for k,(iy,ix,*_) in enumerate(nodes): idx_map[iy,ix]=k
    N = len(nodes)
    coords = np.zeros((N,2), float)
    F = np.zeros((N,8), float)
    intens = np.zeros(N, float)
    for k, (_iy,_ix,gyc,gxc, mean_dst,mean_int,mean_sob,var_lap, ent_loc,mean_H,mean_E,mean_sat) in enumerate(nodes):
        coords[k] = [gyc,gxc]
        intens[k] = mean_int
        F[k,:] = [mean_dst,mean_int,mean_sob,var_lap, ent_loc,mean_H,mean_E,mean_sat]
    rows,cols,vals=[],[],[]
    for k,(iy,ix,*_) in enumerate(nodes):
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            jy,jx = iy+dy, ix+dx
            if jy<0 or jy>=gh or jx<0 or jx>=gw: continue
            j = idx_map[jy,jx]
            if j<0 or j<=k: continue
            dp = coords[k]-coords[j]
            w_pos = math.exp(-(dp@dp)/(2*(DEF_W_POS_SIGMA**2)))
            if DEF_USE_INT_WEIGHT:
                di = intens[k]-intens[j]
                w_int = math.exp(-(di*di)/(2*(DEF_W_INT_SIGMA**2)))
                wij = w_pos * w_int
            else:
                wij = w_pos
            rows += [k,j]; cols += [j,k]; vals += [wij,wij]
    if not rows: return None, None, None
    W = np.zeros((N,N), float); W[rows,cols]=vals; W = 0.5*(W+W.T)
    d = W.sum(axis=1)
    keep = d > 1e-12
    if not np.all(keep):
        W = W[np.ix_(keep, keep)]
        coords = coords[keep]; F = F[keep]
    if W.shape[0] < 3 or W.sum() <= 0:
        return None, None, None
    G = pg.graphs.Graph(W, coords=coords)
    try:
        G.compute_laplacian(lap_type='normalized')
        if hasattr(G.L, "data"): ok = np.isfinite(G.L.data).all()
        else: ok = np.isfinite(G.L).all()
        if not ok: raise ValueError("non-finite L")
    except Exception:
        G.compute_laplacian(lap_type='combinatorial')
    mu = F.mean(axis=0); sd = F.std(axis=0) + 1e-12
    Fz = (F - mu) / sd
    return G, Fz, W

# ====================== spectrum & stabilization (from breakhis_global_binary) ======================

def _safe_partial_spectrum(L, K, which="SM", tol=1e-3, maxiter=5000):
    if not isspmatrix(L): L = csr_matrix(L)
    if L.dtype != np.float64: L = L.asfptype()
    N=L.shape[0]; K=int(max(2, min(K, N-2)))
    try:
        vals, vecs = eigsh(L, k=K, which=which, tol=tol, maxiter=maxiter)
    except Exception:
        try:
            vals, vecs = eigsh(L, k=K, which=which, tol=max(tol,3e-3), maxiter=maxiter*2)
        except Exception:
            if N <= 2000:
                from numpy.linalg import eigh as dense_eigh
                Ld = L.toarray(); vals_all, vecs_all = dense_eigh(Ld)
                order = np.argsort(vals_all)[:K]; vals = vals_all[order]; vecs = vecs_all[:,order]
            else:
                k2 = max(2, K//2)
                vals, vecs = eigsh(L, k=k2, which=which, tol=1e-2, maxiter=maxiter*2)
    order = np.argsort(vals); lam = np.asarray(vals[order]); U = np.asarray(vecs[:,order])
    return lam, U

def _anchor_fields_for_graph(G, featsC):
    N=G.N
    g0 = featsC[:,0].astype(np.float64)
    g0 = (g0 - g0.mean())/(np.linalg.norm(g0)+1e-12)
    if getattr(G, "coords", None) is not None and len(G.coords)==N:
        coords = np.asarray(G.coords, float)
        y = (coords[:,0] - coords[:,0].mean())/(np.linalg.norm(coords[:,0]) + 1e-12)
        x = (coords[:,1] - coords[:,1].mean())/(np.linalg.norm(coords[:,1]) + 1e-12)
    else:
        y = np.zeros(N,float); x = np.zeros(N,float)
    return np.stack([g0,y,x], axis=1)

def _sign_fix_columns(U, anchors):
    U = U.copy(); g0=anchors[:,0]; g1=anchors[:,1]
    dots = U.T @ g0; sgn = np.sign(dots)
    mask0 = (sgn==0)
    if np.any(mask0):
        s1 = np.sign((U[:,mask0].T @ g1))
        sgn[mask0] = np.where(s1==0, 1.0, s1)
    U *= sgn[None,:]
    return U, sgn

def _lambda_bands(lam, n_bands):
    lam = np.asarray(lam, float)
    K = int(lam.size)

    if K <= 0:
        return [np.arange(0, 0, dtype=int) for _ in range(n_bands)]

    if K >= n_bands:
        qs = np.linspace(0.0, 1.0, n_bands + 1)
        cuts = np.searchsorted(lam, np.quantile(lam, qs, method="linear"))
        cuts[0] = 0
        cuts[-1] = K
        cuts = np.clip(cuts, 0, K)
        return [np.arange(cuts[i], cuts[i + 1]) for i in range(n_bands)]

    edges = np.linspace(0, K, n_bands + 1)
    edges = np.floor(edges).astype(int)
    edges[-1] = K
    edges = np.clip(edges, 0, K)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_bands)]



def _band_align(U, a_mat, lam, anchors, n_bands):
    U2 = U.copy(); a2 = a_mat.copy(); bands = _lambda_bands(lam, n_bands); A=anchors
    for idx in bands:
        UB = U2[:, idx]; S = UB.T @ A
        try: R,_,_ = np.linalg.svd(S, full_matrices=False)
        except np.linalg.LinAlgError: R = np.eye(idx.size)
        U2[:, idx] = UB @ R
        a2[:, idx] = a2[:, idx] @ R
    return U2, a2

def _band_energies(a_mat, lam, n_bands, eps=1e-12, l2norm=True):
    bands = _lambda_bands(lam, n_bands); C,K = a_mat.shape
    E = np.zeros((C, len(bands)), np.float32)
    for j, idx in enumerate(bands):
        E[:,j] = np.sum(a_mat[:, idx]**2, axis=1)
    if l2norm:
        denom = np.linalg.norm(E, axis=1, keepdims=True) + eps
    else:
        denom = E.sum(axis=1, keepdims=True) + eps
    return (E/denom).astype(np.float32)

# ---------- FRACTAL ENCODER (spectral-axis) ----------

def _fit_loglog_slope(y, eps=1e-12):
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
    p = np.asarray(p, float)
    p = p / (p.sum() + eps)
    B = int(p.size)
    out = []
    logB = np.log(B + eps)
    H = -np.sum(p * np.log(p + eps))  # Shannon
    for q in qs:
        if abs(q - 1.0) < 1e-12:
            out.append(float(H / (logB + eps)))
        else:
            z = np.sum(np.power(p + eps, q))
            out.append(float(np.log(z + eps) / ((q - 1.0) * (logB + eps))))
    return out

def _lacunarity_1d(y, windows=(2, 4, 8), eps=1e-12):
    y = np.asarray(y, float)
    y = y - np.min(y) + eps
    K = int(y.size)
    vals = []
    for w in windows:
        if w > K:
            vals.append(1.0); continue
        cums = np.cumsum(np.r_[0.0, y])
        box = cums[w:] - cums[:-w]
        mu = float(np.mean(box))
        var = float(np.var(box))
        Λ = var / (mu*mu + eps) + 1.0
        vals.append(float(Λ))
    return vals

def fractal_encode_channels(E, eps=1e-12):
    E = np.asarray(E, float)
    C, K = E.shape
    colsum = np.sum(E, axis=0)
    k_eff = int(np.max(np.nonzero(colsum > eps)[0])+1) if np.any(colsum > eps) else 0
    if k_eff >= 6:
        E = E[:, :k_eff]
    feats = []
    for c in range(C):
        y = E[c]
        if y.size < 6 or not np.isfinite(y).all() or np.all(y <= eps):
            feats += [0.0, 0.0] + [0.0, 0.0, 0.0, 0.0] + [1.0, 1.0, 1.0]
            continue
        slope, r2 = _fit_loglog_slope(y, eps=eps)
        p = y / (y.sum() + eps)
        Dq = _renyi_dimensions(p, qs=(0.5, 1.0, 2.0, 4.0), eps=eps)
        lac = _lacunarity_1d(y, windows=(2,4,8), eps=eps)
        feats += [slope, r2] + Dq + lac
    return np.array(feats, dtype=np.float32)

# ---------- core spectral coefficients ----------

def pygsp_spectral_coeffs(G, featsC, k_reg=DEF_K_REG):
    if not hasattr(G,"L") or G.L is None: G.compute_laplacian(lap_type="normalized")
    N=int(G.N); K=int(max(2, min(k_reg, N-2)))
    lam, U = _safe_partial_spectrum(G.L, K, which="SM", tol=1e-3, maxiter=5000)
    X = featsC.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
    a_mat = (U.T @ X).T  # (C,K)
    anchors = _anchor_fields_for_graph(G, featsC)
    U_sf, sgn = _sign_fix_columns(U, anchors)
    a_mat *= sgn[None,:]
    return lam, U_sf, a_mat, anchors

# ====================== extra spectral features (from breakhis_global_binary) ======================

def hks_features(U: np.ndarray, lam: np.ndarray, t_scales: Tuple[float,...]=DEF_HKS_T,
                 pool: str = "meanstd") -> np.ndarray:
    U = np.asarray(U, float)
    lam = np.asarray(lam, float)
    Phi2 = U**2  # (N,K)
    feats=[]
    for t in t_scales:
        hks = Phi2 @ np.exp(-t*lam)  # (N,)
        if pool == "meanstd":
            feats += [float(hks.mean()), float(hks.std())]
        else:
            feats += [float(hks.mean())]
    return np.array(feats, dtype=np.float32)

def slepian_concentration_features(G: 'pg.graphs.Graph', U: np.ndarray, lam: np.ndarray,
                                   anchors: np.ndarray, n_vecs: int = 32, frac: float = 0.5) -> np.ndarray:
    N = U.shape[0]
    if N == 0: return np.zeros(3, np.float32)
    roi_mask = anchors[:,0] > np.quantile(anchors[:,0], 1.0-frac)
    A = np.diag(roi_mask.astype(float))
    k = int(min(n_vecs, U.shape[1]))
    conc = []
    for j in range(k):
        uj = U[:,j:j+1]
        num = float((uj.T @ A @ uj)[0,0])
        den = float((uj.T @ uj)[0,0]) + 1e-12
        conc.append(num / den)
    conc = np.array(conc, float)
    top5 = np.sort(conc)[-5:] if conc.size>=5 else conc
    return np.array([float(conc.mean()), float(conc.std()), float(top5.mean())], dtype=np.float32)

def wavelet_energy_features(G: 'pg.graphs.Graph', X: np.ndarray,
                            t_scales: Tuple[float,...]=DEF_WAVE_T) -> np.ndarray:
    if not HAS_PYGSP or G is None or X is None or X.size == 0:
        return np.zeros((X.shape[1]*len(t_scales),), np.float32) if X is not None else np.zeros(0, np.float32)
    L = G.L if isspmatrix(G.L) else csr_matrix(G.L)
    N = L.shape[0]
    C = X.shape[1]
    out = []
    k_eigs = min(max(10, int(0.8*N)), 256) if N > 20 else max(2, N-2)
    try:
        vals, vecs = eigsh(L, k=k_eigs, which='SM', tol=1e-3, maxiter=5000)
    except Exception:
        from numpy.linalg import eigh as dense_eigh
        Ld = L.toarray(); vals, vecs = dense_eigh(Ld)
    Ue = np.asarray(vecs); lam = np.asarray(vals)
    for c in range(C):
        f = X[:,c:c+1]
        a = Ue.T @ f
        e_scales=[]
        for t in t_scales:
            y = Ue @ (np.exp(-t*lam)[:,None] * a)
            e = float(np.linalg.norm(y)**2)
            e_scales.append(e)
        e_scales = np.array(e_scales, float)
        e_scales = e_scales / (e_scales.sum() + 1e-12)
        out.append(e_scales)
    return np.concatenate(out, axis=0).astype(np.float32)

def persistent_features(coords: np.ndarray, max_dim: int = 1) -> np.ndarray:
    if not HAS_GUDHI or coords is None or coords.shape[0] < 5:
        return np.zeros(5, dtype=np.float32)
    rc = gudhi.RipsComplex(points=coords)
    st = rc.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()
    out = []
    for dim in (0,1):
        diag = st.persistence_intervals_in_dimension(dim)
        if diag is None or len(diag)==0:
            if dim==0:
                out += [0.0, 0.0]
            else:
                out += [0.0, 0.0, 0.0]
            continue
        lifetimes = [float(bd[1]-bd[0]) for bd in diag if np.isfinite(bd[1])]
        if dim==0:
            out += [float(len(lifetimes)), float(np.sum(lifetimes))]
        else:
            s = float(np.sum(lifetimes))
            p = np.array(lifetimes, float)
            p = p / (p.sum() + 1e-12)
            ent = float(-(p[p>0]*np.log(p[p>0])).sum())
            out += [float(len(lifetimes)), s, ent]
    return np.array(out, dtype=np.float32)

# ====================== sequences & FULL vector (from breakhis_global_binary) ======================

def sequences_for_path_pygsp(path: str, cfg: Dict, spectral_mode: str = "invariant", n_bands: int = 8):
    rgb = read_rgb(path)
    if rgb is None: return None
    mask, gray = tissue_mask_largest(rgb, min_area_frac=cfg["cell_thr"])
    if mask is None: return None
    G, featsC, _ = region_graph_pygsp(rgb, gray, mask,
                                      grid_n=cfg["grid_n"], cell_thr=cfg["cell_thr"],
                                      pos_sigma=cfg["pos_sigma"], int_sigma=cfg["int_sigma"],
                                      use_int_weight=cfg["use_int_weight"])
    if G is None or featsC is None or featsC.size==0: return None
    lam, U_sf, a_mat, anchors = pygsp_spectral_coeffs(G, featsC, k_reg=cfg["k_reg"])
    if spectral_mode == "raw":
        Kf = int(cfg["k_reg"])  # pad/truncate to Kf
        out = np.zeros((a_mat.shape[0], Kf), np.float32); n=min(Kf, a_mat.shape[1])
        out[:, :n] = a_mat[:, :n].astype(np.float32); return out
    if spectral_mode == "aligned":
        _, a_al = _band_align(U_sf, a_mat, lam, anchors, n_bands)
        Kf = int(cfg["k_reg"])  # pad/truncate to Kf
        out = np.zeros((a_al.shape[0], Kf), np.float32); n=min(Kf, a_al.shape[1])
        out[:, :n] = a_al[:, :n].astype(np.float32); return out
    if spectral_mode == "invariant":
        E = _band_energies(a_mat.astype(np.float32), lam, n_bands, l2norm=True)
        return E.astype(np.float32)
    return None

def full_feature_vector(G, featsC, lam, U_sf, a_mat, anchors, bands:int) -> np.ndarray:
    """Assemble the 'FULL' pooled feature vector used by LGBM/LR tracks."""
    # invariant band energies (per-channel across bands)
    E = _band_energies(a_mat.astype(np.float32), lam, bands, l2norm=True)  # (C,Kb)
    # base pooled stats over spectral axis
    base_pool = []
    for c in range(E.shape[0]):
        v = E[c]
        base_pool += [float(v.mean()), float(v.std())]
    base_pool = np.array(base_pool, np.float32)  # 2C
    # fractal on band energies
    frac = fractal_encode_channels(E)  # 9*C
    # HKS pooled on nodes
    hks = hks_features(U_sf, lam, t_scales=DEF_HKS_T, pool="meanstd")  # 2*len(T)
    # Slepian concentration
    slep = slepian_concentration_features(G, U_sf, lam, anchors, n_vecs=min(32, U_sf.shape[1]), frac=0.5)  # 3
    # Graph spectral wavelet energy per channel/scale on node features
    wave = wavelet_energy_features(G, featsC, t_scales=DEF_WAVE_T)  # C*len(T)
    # Persistent homology on region centroids
    coords = np.asarray(G.coords, float) if getattr(G, "coords", None) is not None else None
    pers = persistent_features(coords, max_dim=1)  # 5
    # concat
    parts = [base_pool, frac, hks, wave, slep, pers]
    vec = np.concatenate([p.astype(np.float32) for p in parts if p is not None and p.size>0], axis=0)
    # safety
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vec


def compute_all_for_patch(path: str, cfg: Dict, spectral_mode: str = "invariant", n_bands: int = 8):
    """Return (sequence CxK, full_features, rgb) for a patch image path."""
    rgb = read_rgb(path)
    if rgb is None: return None, None, None
    mask, gray = tissue_mask_largest(rgb, min_area_frac=cfg["cell_thr"])
    if mask is None: return None, None, None
    G, featsC, _ = region_graph_pygsp(rgb, gray, mask,
                                      grid_n=cfg["grid_n"], cell_thr=cfg["cell_thr"],
                                      pos_sigma=cfg["pos_sigma"], int_sigma=cfg["int_sigma"],
                                      use_int_weight=cfg["use_int_weight"])
    if G is None or featsC is None or featsC.size==0: return None, None, None
    lam, U_sf, a_mat, anchors = pygsp_spectral_coeffs(G, featsC, k_reg=cfg["k_reg"])
    # sequence according to requested mode
    if spectral_mode == "raw":
        Kf = int(cfg["k_reg"])  # pad/truncate to Kf
        out = np.zeros((a_mat.shape[0], Kf), np.float32); n=min(Kf, a_mat.shape[1]); out[:,:n]=a_mat[:,:n].astype(np.float32)
        Xseq = out
    elif spectral_mode == "aligned":
        _, a_al = _band_align(U_sf, a_mat, lam, anchors, n_bands)
        Kf = int(cfg["k_reg"])  # pad/truncate to Kf
        out = np.zeros((a_al.shape[0], Kf), np.float32); n=min(Kf, a_al.shape[1]); out[:,:n]=a_al[:,:n].astype(np.float32)
        Xseq = out
    else:
        Xseq = _band_energies(a_mat.astype(np.float32), lam, n_bands, l2norm=True).astype(np.float32)
    Xfull = full_feature_vector(G, featsC, lam, U_sf, a_mat, anchors, bands=n_bands)
    return Xseq, Xfull, rgb

# ====================== baselines & splits ======================

def sanitize_X(Xtr, Xte, var_thresh=1e-10):
    keep = np.isfinite(Xtr).all(0) & (Xtr.std(0) > var_thresh)
    if not keep.any(): keep = np.isfinite(Xtr).all(0)
    return Xtr[:,keep], Xte[:,keep], keep

def stratified_group_splits(labels, groups, n_splits=DEF_FOLDS, seed=DEF_SEED):
    idx=np.arange(len(labels))
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(idx, labels, groups))
    # fallback: approximate via stratified on majority class per group
    uniq = np.array(sorted(list(set(groups))))
    maj=[]
    for u in uniq:
        labs = np.array(labels)[np.array(groups)==u]
        vals,cnts = np.unique(labs, return_counts=True)
        maj.append(vals[np.argmax(cnts)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits=[]
    for _, te_g in skf.split(uniq, maj):
        te_pat=set(uniq[te_g])
        te=[i for i,g in enumerate(groups) if g in te_pat]
        tr=[i for i in range(len(groups)) if i not in te]
        splits.append((np.array(tr), np.array(te)))
    return splits

# ====================== CNN embedding (for baseline + fusion) ======================
_CNN_MODEL = None
_CNN_TX = None
_CNN_DEVICE = None

def _init_cnn_model():
    global _CNN_MODEL, _CNN_TX, _CNN_DEVICE
    if not (HAS_TORCH and HAS_TORCHVISION):
        return False
    if _CNN_MODEL is None:
        try:
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        except Exception:
            m = tvm.resnet18(weights=None)
        m.fc = nn.Identity()  # 512-D output
        _CNN_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = m.to(_CNN_DEVICE).eval()
        _CNN_MODEL = m
        _CNN_TX = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    return True

def cnn_embed_image(rgb: np.ndarray) -> np.ndarray:
    if not _init_cnn_model():
        return np.zeros(512, np.float32)
    h,w = rgb.shape[:2]
    cy,cx = h//2, w//2
    L = min(h,w)
    y0 = max(0, cy - L//2); y1 = y0 + L
    x0 = max(0, cx - L//2); x1 = x0 + L
    patch = rgb[y0:y1, x0:x1].copy()
    x = _CNN_TX(patch).unsqueeze(0).to(_CNN_DEVICE)
    with torch.no_grad():
        z = _CNN_MODEL(x).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (512,)
    return z

def load_or_build_cnn_embeddings(paths: np.ndarray, out_dir: Path, tag: str = "gbm") -> np.ndarray:
    cache_path = Path(out_dir) / f"cnn_embed_{tag}.npz"
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            E = data["E"]; P = data["paths"]
            if len(P)==len(paths) and np.all(P==paths):
                log.info(f"[cache] loaded {cache_path}")
                return E.astype(np.float32)
        except Exception:
            pass
    Emat = []
    it = tqdm(paths, desc=f"[CNNemb] {tag}")
    for p in it:
        rgb = read_rgb(p)
        if rgb is None:
            Emat.append(np.zeros(512, np.float32))
        else:
            Emat.append(cnn_embed_image(rgb))
    E = np.stack(Emat, axis=0).astype(np.float32)
    try:
        np.savez_compressed(cache_path, E=E, paths=np.array(paths))
        log.info(f"[cache] saved {cache_path}")
    except Exception as e:
        log.info(f"[warn] could not save CNN cache: {e}")
    return E

# ====================== SNO ======================
if HAS_TORCH:
    class SpectralMix(nn.Module):
        def __init__(self, C, K, rank=None, nonlin='softplus'):
            super().__init__()
            self.C,self.K=C,K
            self.nonlin = nn.Softplus() if nonlin=='softplus' else nn.Tanh()
            if rank is None or rank>=K:
                self.rank=None; self.A = nn.Linear(K,K, bias=False)
            else:
                self.rank=int(rank); self.A1=nn.Linear(K,self.rank,bias=False); self.A2=nn.Linear(self.rank,K,bias=False)
            self.B = nn.Linear(C,C, bias=False)
            self.bias = nn.Parameter(torch.zeros(1,C,K))
        def forward(self, U):
            assert U.dim()==3, f"SpectralMix expects (B,C,K), got {tuple(U.shape)}"
            Bsz,C,K=U.shape
            if K!=self.K:
                if K<self.K: U=torch.cat([U, torch.zeros(Bsz,C,self.K-K, device=U.device, dtype=U.dtype)], dim=-1)
                else: U=U[...,:self.K]
                K=self.K
            if C!=self.C:
                if C<self.C: U=torch.cat([U, torch.zeros(Bsz,self.C-C,K, device=U.device, dtype=U.dtype)], dim=1)
                else: U=U[:,:self.C,:]
                C=self.C
            X = U.reshape(Bsz*C, K)
            X = self.A(X) if self.rank is None else self.A2(self.A1(X))
            UA = X.reshape(Bsz, C, K)
            BU = self.B(UA.transpose(1,2)).transpose(1,2)
            return self.nonlin(BU + self.bias)

    class SNOClassifier(nn.Module):
        def __init__(self, C=8, K=128, hidden=64, rank=32, pdrop=0.1, n_classes: int = 2):
            super().__init__()
            self.mix1=SpectralMix(C,K,rank=rank, nonlin='softplus')
            self.mix2=SpectralMix(C,K,rank=rank, nonlin='softplus')
            self.drop=nn.Dropout(pdrop)
            self.pool=nn.AdaptiveAvgPool1d(1)
            self.head=nn.Sequential(nn.Flatten(), nn.Linear(C, hidden), nn.Softplus(), nn.Dropout(pdrop), nn.Linear(hidden,n_classes))
        def forward(self, U):
            x=self.mix1(U); x=self.drop(x); x=self.mix2(x); z=self.pool(x).squeeze(-1); return self.head(z)
        def forward_with_feat(self, U):
            x=self.mix1(U); x=self.drop(x); x=self.mix2(x); z=self.pool(x).squeeze(-1)
            logits=self.head(z); return logits, z

def _macro_auc_ovr(y_true_int: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    # If the test fold has only one class present, macro AUC is undefined.
    if len(np.unique(y_true_int)) < 2:
        return float('nan')
    try:
        yb = label_binarize(y_true_int, classes=list(range(n_classes)))
        return float(roc_auc_score(yb, proba, average='macro', multi_class='ovr'))
    except Exception:
        return float('nan')


def train_eval_sno_cv(Xseq, y_int, splits, args, n_classes: int):
    """Train SNO per fold; return metrics and per-fold pooled embeddings for LGBM + fusion."""
    if not HAS_TORCH or Xseq is None or len(Xseq)==0:
        log.info("PyTorch not available or no SNO sequences; skipping SNO."); return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N,C,K = Xseq.shape; log.info(f"SNO: N={N}, C={C}, K={K}, device={device}, classes={n_classes}")
    sno_accs,sno_bals,sno_aucs=[],[],[]
    fold_feats=[]  # list of dicts with tr, te, Ztr, Zte, ytr, yte
    for fold,(tr,te) in enumerate(splits,1):
        Utr=torch.from_numpy(Xseq[tr]).to(device); Ute=torch.from_numpy(Xseq[te]).to(device)
        ytr_t=torch.from_numpy(y_int[tr].astype(np.int64)).to(device); yte_t=torch.from_numpy(y_int[te].astype(np.int64)).to(device)
        model=SNOClassifier(C=C, K=K, hidden=args.sno_hidden,
                            rank=min(args.sno_rank, K//2) if args.sno_rank>0 else None,
                            pdrop=args.sno_dropout, n_classes=n_classes).to(device)
        opt=torch.optim.AdamW(model.parameters(), lr=args.sno_lr, weight_decay=args.sno_wd)
        # class weights inverse frequency
        wts = torch.ones(n_classes, dtype=torch.float32, device=device)
        for c in range(n_classes):
            cnt = (ytr_t==c).sum().item()
            if cnt>0:
                wts[c] = float(len(ytr_t))/float(n_classes*cnt)
        criterion=nn.CrossEntropyLoss(weight=wts)
        model.train(); idx_all=torch.arange(Utr.size(0), device=device)
        for _ in range(args.sno_epochs):
            idx=idx_all[torch.randperm(idx_all.numel())]
            for s in range(0,len(idx), args.sno_batch):
                b=idx[s:s+args.sno_batch]; logits=model(Utr[b]); loss=criterion(logits, ytr_t[b])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(Ute), dim=1).detach().cpu().numpy()  # (n_te, n_classes)
            # pooled features for LGBM/fusion (train & test for this fold)
            _, z_tr = model.forward_with_feat(Utr)
            _, z_te = model.forward_with_feat(Ute)
            Ztr = z_tr.detach().cpu().numpy().astype(np.float32)
            Zte = z_te.detach().cpu().numpy().astype(np.float32)
        pred = prob.argmax(axis=1)
        acc=accuracy_score(y_int[te], pred); bal=balanced_accuracy_score(y_int[te], pred)
        auc=_macro_auc_ovr(y_int[te], prob, n_classes)
        log.info(f"[SNO] Fold {fold}: Acc {acc:.3f} | BalAcc {bal:.3f} | mAUC {auc:.3f}")
        sno_accs.append(acc); sno_bals.append(bal); sno_aucs.append(auc)
        fold_feats.append(dict(tr=tr, te=te, Ztr=Ztr, Zte=Zte, ytr=y_int[tr], yte=y_int[te]))
    log.info(f"\n*** SNO MEAN over folds ***")
    log.info(f"Accuracy: {np.mean(sno_accs):.3f} ± {np.std(sno_accs):.3f}")
    log.info(f"Balanced Acc: {np.mean(sno_bals):.3f} ± {np.std(sno_bals):.3f}")
    log.info(f"mAUC (OvR): {np.nanmean(sno_aucs):.3f} ± {np.nanstd(sno_aucs):.3f}")
    return dict(acc=sno_accs, bal=sno_bals, auc=sno_aucs, folds=fold_feats)

# ====================== fusion utils ======================

def group_permutation_importance_multiclass(clf, Xte, yte_int, groups: dict, n_classes: int, seed=DEF_SEED):
    rng = np.random.default_rng(seed)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(Xte)
        if isinstance(prob, list):  # some libs return list per class
            prob = np.stack([p[:,1] if p.ndim==2 else p for p in prob], axis=1)
    else:
        # use decision_function; map to pseudo-proba via softmax
        df = clf.decision_function(Xte)
        if df.ndim==1:
            df = np.stack([-df, df], axis=1)
        e = np.exp(df - df.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
    base = _macro_auc_ovr(yte_int, prob, n_classes)
    out = {'base_mAUC': float(base)}
    for name, idxs in groups.items():
        Xp = Xte.copy()
        for j in idxs:
            rng.shuffle(Xp[:, j])
        if hasattr(clf, "predict_proba"):
            prob2 = clf.predict_proba(Xp)
            if isinstance(prob2, list):
                prob2 = np.stack([p[:,1] if p.ndim==2 else p for p in prob2], axis=1)
        else:
            df = clf.decision_function(Xp)
            if df.ndim==1:
                df = np.stack([-df, df], axis=1)
            e = np.exp(df - df.max(axis=1, keepdims=True))
            prob2 = e / e.sum(axis=1, keepdims=True)
        auc2 = _macro_auc_ovr(yte_int, prob2, n_classes)
        out[name] = float(base - auc2) if np.isfinite(auc2) else float('nan')
    return out

# ====================== run dataset ======================

def run_dataset(meta: List[Dict], out_dir: Path, cfg, args):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"g{cfg['grid_n']}_cell{int(cfg['cell_thr']*100)}_K{cfg['k_reg']}_{args.spectral_mode}_B{args.bands}"
    cache_seq = Path(out_dir) / f"snoseq_GBM_{tag}.npz"
    cache_full = Path(out_dir) / f"fullfeat_GBM_{tag}.npz"

    Xseq_all=None; y_all=None; p_all=None; Xfull_all=None; img_group=None

    # ----- load or build caches -----
    load_ok = False
    if cache_seq.exists() and cache_full.exists() and not args.rebuild:
        try:        
            data_seq = np.load(cache_seq, allow_pickle=False)
            data_full = np.load(cache_full, allow_pickle=False)
            Xseq_all = data_seq["Xseq"].astype(np.float32)
            p_all = data_seq["paths"].astype(str)
            y_all = data_seq["y"].astype(str)
            img_group = data_seq["groups"].astype(str)
            Xfull_all = data_full["Xfull"].astype(np.float32)
            load_ok = True
            log.info(f"[cache] loaded sequences + FULL features from {cache_seq.name} / {cache_full.name}")
        except Exception as e:
            log.info(f"[cache] failed to load caches ({e}); rebuilding.")
            load_ok = False

    if not load_ok:
        paths = []
        labels = []
        groups = []
        Xseq_list = []
        Xfull_list = []

        log.info(f"[build] computing spectral sequences + FULL features for {len(meta)} patches...")
        for rec in tqdm(meta, desc="[build] patches"):
            p = rec["patch_path"]
            cls = rec["region"]
            grp = rec["image"]
            seq, fullv, rgb = compute_all_for_patch(
                p,
                cfg,
                spectral_mode=args.spectral_mode,
                n_bands=args.bands
            )
            if seq is None or fullv is None:
                continue
            # === FIXED SHAPE FOR STACK ===
            C_EXP = 8  # our node feature channels in region_graph_pygsp
            K_EXP = args.bands if args.spectral_mode == "invariant" else args.k_reg

            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# Ensure 2D
            if seq.ndim == 1:
                seq = seq.reshape(C_EXP, -1)

# Pad/trim to (C_EXP, K_EXP)
            C_now = min(seq.shape[0], C_EXP)
            K_now = min(seq.shape[1], K_EXP)
            seq_fixed = np.zeros((C_EXP, K_EXP), dtype=np.float32)
            seq_fixed[:C_now, :K_now] = seq[:C_now, :K_now]
            seq = seq_fixed
# === END FIX ===

            Xseq_list.append(seq.astype(np.float32))
            Xfull_list.append(fullv.astype(np.float32))
            paths.append(p)
            labels.append(cls)
            groups.append(grp)

        if not Xseq_list:
            raise RuntimeError("No valid patches after feature computation.")

        # sequences are variable CxK, but our code returns fixed K via padding/truncation above
        Xseq_all = np.stack(Xseq_list, axis=0).astype(np.float32)   # (N, C, K)
        Xfull_all = np.stack(Xfull_list, axis=0).astype(np.float32) # (N, D_full)
        y_all = np.array(labels)
        p_all = np.array(paths)
        img_group = np.array(groups)

        # cache
        np.savez_compressed(cache_seq, Xseq=Xseq_all, paths=p_all, y=y_all, groups=img_group)
        np.savez_compressed(cache_full, Xfull=Xfull_all)
        log.info(f"[cache] wrote {cache_seq.name} and {cache_full.name}")

    # ----- class encoding -----
    classes = sorted(list(np.unique(y_all)))
    if args.exclude_background and "Background" in classes:
        keep = y_all != "Background"
        Xseq_all = Xseq_all[keep]
        Xfull_all = Xfull_all[keep]
        p_all = p_all[keep]
        img_group = img_group[keep]
        y_all = y_all[keep]
        classes = sorted(list(np.unique(y_all)))
        log.info(f"[filter] excluded 'Background'. N={len(y_all)}, classes={classes}")

    n_classes = len(classes)
    cls_to_int = {c:i for i,c in enumerate(classes)}
    y_int = np.array([cls_to_int[c] for c in y_all], dtype=np.int64)

    # optional cap
    if args.max_n is not None and args.max_n > 0 and len(y_int) > args.max_n:
        idx = np.random.default_rng(args.seed).choice(len(y_int), size=args.max_n, replace=False)
        idx = np.sort(idx)
        Xseq_all = Xseq_all[idx]
        Xfull_all = Xfull_all[idx]
        y_int = y_int[idx]
        y_all = y_all[idx]
        p_all = p_all[idx]
        img_group = img_group[idx]
        log.info(f"[cap] subsampled to N={len(y_int)}")

    # ----- CNN embeddings (baseline + fusion) -----
    E_cnn = load_or_build_cnn_embeddings(p_all, out_dir, tag="GBM")

    # ----- CV splits (grouped) -----
    splits = stratified_group_splits(y_int, img_group, n_splits=args.folds, seed=args.seed)
    log.info(f"[CV] {len(splits)} folds | grouped by image | classes={classes}")

    # ============================ 5 tracks ============================
    results = {}

    # A) SNO training (also yields pooled embeddings per fold)
    sno_cv = train_eval_sno_cv(
        Xseq=Xseq_all,
        y_int=y_int,
        splits=splits,
        args=args,
        n_classes=n_classes
    )
    results["SNO"] = sno_cv

    # helper: metric logger
    def _eval_multiclass_model(model_name, y_true, prob_pred):
        pred = np.argmax(prob_pred, axis=1)
        acc = accuracy_score(y_true, pred)
        bal = balanced_accuracy_score(y_true, pred)
        auc = _macro_auc_ovr(y_true, prob_pred, n_classes)
        log.info(f"[{model_name}] Acc {acc:.3f} | BalAcc {bal:.3f} | mAUC {auc:.3f}")
        return acc, bal, auc

    # helper: LGBM factory
    def _lgbm_mc():
        if not HAS_LGBM:
            return None
        return LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=args.lgbm_estimators,
            learning_rate=args.lgbm_lr,
            max_depth=args.lgbm_max_depth,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample,
            reg_lambda=args.lgbm_l2,
            reg_alpha=args.lgbm_l1,
            class_weight="balanced" if (not HAS_IMB or not args.oversample) else None,
            random_state=args.seed,
            n_jobs=args.n_jobs
        )

    # B) Baseline CNN → LGBM
    if HAS_LGBM:
        log.info("\n==== (1) Baseline CNN → LGBM ====")
        accs, bals, aucs = [], [], []
        for fi, (tr, te) in enumerate(splits, 1):
            Xtr, Xte = E_cnn[tr], E_cnn[te]
            ytr, yte = y_int[tr], y_int[te]
            if HAS_IMB and args.oversample:
                ros = RandomOverSampler(random_state=args.seed)
                Xtr, ytr = ros.fit_resample(Xtr, ytr)
            clf = _lgbm_mc()
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)
            a,b,auc = _eval_multiclass_model("CNN→LGBM", yte, prob)
            accs.append(a); bals.append(b); aucs.append(auc)
        results["CNN_LGBM"] = dict(acc=accs, bal=bals, auc=aucs)
    else:
        log.info("[skip] lightgbm not available; skipping CNN→LGBM")

    # C) Spectral pooled (FULL) → LGBM  (+ permutation groups)
    if HAS_LGBM:
        log.info("\n==== (3) FULL pooled → LGBM ====")
        accs, bals, aucs = [], [], []
        perm_by_fold = []
        # group indices for permutation importance (based on construction in full_feature_vector)
        C = 8
        sz_base = 2*C
        sz_frac = 9*C
        sz_hks  = 2*len(DEF_HKS_T)
        sz_wave = C*len(DEF_WAVE_T)
        sz_slep = 3
        sz_pers = 5
        starts = np.cumsum([0, sz_base, sz_frac, sz_hks, sz_wave, sz_slep])
        grp_idx = {
            "base_pool": list(range(starts[0], starts[1])),
            "fractal":   list(range(starts[1], starts[2])),
            "hks":       list(range(starts[2], starts[3])),
            "wave":      list(range(starts[3], starts[4])),
            "slepian":   list(range(starts[4], starts[5])),
            "persistent":list(range(starts[5], starts[5]+sz_pers)),
        }
        for fi, (tr, te) in enumerate(splits, 1):
            Xtr, Xte = Xfull_all[tr], Xfull_all[te]
            ytr, yte = y_int[tr], y_int[te]
            if HAS_IMB and args.oversample:
                ros = RandomOverSampler(random_state=args.seed)
                Xtr, ytr = ros.fit_resample(Xtr, ytr)
            clf = _lgbm_mc()
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)
            a,b,auc = _eval_multiclass_model("FULL→LGBM", yte, prob)
            accs.append(a); bals.append(b); aucs.append(auc)
            # permutation (only on first fold unless --perm_all)
            if fi == 1 or args.perm_all_folds:
                perm = group_permutation_importance_multiclass(clf, Xte, yte, grp_idx, n_classes, seed=args.seed)
                perm_by_fold.append(dict(fold=fi, importance=perm))
        results["FULL_LGBM"] = dict(acc=accs, bal=bals, auc=aucs, perm=perm_by_fold)
    else:
        log.info("[skip] lightgbm not available; skipping FULL→LGBM")

    # D) Spectral pooled (FULL) → LogisticRegression L2
    log.info("\n==== (4) FULL pooled → LogisticRegression (L2) ====")
    accs, bals, aucs = [], [], []
    for fi, (tr, te) in enumerate(splits, 1):
        Xtr, Xte = Xfull_all[tr], Xfull_all[te]
        ytr, yte = y_int[tr], y_int[te]
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        lr = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            multi_class="ovr",
            class_weight="balanced",
            n_jobs=args.n_jobs
        )
        lr.fit(Xtr_s, ytr)
        prob = lr.predict_proba(Xte_s)
        a,b,auc = _eval_multiclass_model("FULL→LR", yte, prob)
        accs.append(a); bals.append(b); aucs.append(auc)
    results["FULL_LR"] = dict(acc=accs, bal=bals, auc=aucs)

    # E) SNO pooled embedding → LGBM
    if HAS_LGBM and sno_cv is not None:
        log.info("\n==== (5) SNO pooled → LGBM ====")
        accs, bals, aucs = [], [], []
        for fi, fold in enumerate(sno_cv["folds"], 1):
            Ztr, Zte = fold["Ztr"], fold["Zte"]
            ytr, yte = fold["ytr"], fold["yte"]
            if HAS_IMB and args.oversample:
                ros = RandomOverSampler(random_state=args.seed)
                Ztr, ytr = ros.fit_resample(Ztr, ytr)
            clf = _lgbm_mc()
            clf.fit(Ztr, ytr)
            prob = clf.predict_proba(Zte)
            a,b,auc = _eval_multiclass_model("SNO→LGBM", yte, prob)
            accs.append(a); bals.append(b); aucs.append(auc)
        results["SNO_LGBM"] = dict(acc=accs, bal=bals, auc=aucs)
    else:
        log.info("[skip] SNO pooled → LGBM (missing lightgbm or SNO stage)")

    # F) CNN ⊕ SNO fusion (PCA + Hadamard) → LGBM
    if HAS_LGBM and sno_cv is not None:
        log.info("\n==== (2) CNN ⊕ SNO (PCA + Hadamard) → LGBM ====")
        accs, bals, aucs = [], [], []
        for fi, (tr, te) in enumerate(splits, 1):
            # collect pooled SNO features aligned to the current split
            fold = sno_cv["folds"][fi-1]
            Ztr, Zte = fold["Ztr"], fold["Zte"]
            ytr, yte = fold["ytr"], fold["yte"]

            Etr, Ete = E_cnn[tr], E_cnn[te]

            # Standardize each modality then PCA to same d
            sc_cnn = StandardScaler().fit(Etr)
            sc_sno = StandardScaler().fit(Ztr)
            Etr_s = sc_cnn.transform(Etr); Ete_s = sc_cnn.transform(Ete)
            Ztr_s = sc_sno.transform(Ztr); Zte_s = sc_sno.transform(Zte)

            d_max = min(args.fusion_pca_dim, Etr_s.shape[1], Ztr_s.shape[1], Etr_s.shape[0]-1, Ztr_s.shape[0]-1)
            d = max(2, d_max)
            pca_c = PCA(n_components=d, random_state=args.seed).fit(Etr_s)
            pca_s = PCA(n_components=d, random_state=args.seed).fit(Ztr_s)
            Ec_tr = pca_c.transform(Etr_s); Ec_te = pca_c.transform(Ete_s)
            Zc_tr = pca_s.transform(Ztr_s); Zc_te = pca_s.transform(Zte_s)

            H_tr = Ec_tr * Zc_tr
            H_te = Ec_te * Zc_te
            F_tr = np.concatenate([Ec_tr, Zc_tr, H_tr], axis=1)
            F_te = np.concatenate([Ec_te, Zc_te, H_te], axis=1)

            if HAS_IMB and args.oversample:
                ros = RandomOverSampler(random_state=args.seed)
                F_tr, ytr = ros.fit_resample(F_tr, ytr)

            clf = _lgbm_mc()
            clf.fit(F_tr, ytr)
            prob = clf.predict_proba(F_te)
            a,b,auc = _eval_multiclass_model("CNN⊕SNO→LGBM", yte, prob)
            accs.append(a); bals.append(b); aucs.append(auc)
        results["FUSION_CNN_SNO_LGBM"] = dict(acc=accs, bal=bals, auc=aucs)
    else:
        log.info("[skip] fusion (missing lightgbm or SNO stage)")

    # ----- persist results -----
    with open(Path(out_dir) / f"metrics_{tag}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    log.info(f"[done] wrote metrics to metrics_{tag}.json")

    return results


# ====================== CLI ======================

def parse_args():
    p = argparse.ArgumentParser(description="GBM multi-class global pipeline (FULL features + SNO + CNN baselines)")
    # data
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--patch_dir", type=str, default=None)
    p.add_argument("--build_patches", action="store_true", help="extract contour patches from image/mask dirs")
    p.add_argument("--labels_json", type=str, default=None, help="override GBM label colors mapping JSON")
    p.add_argument("--exclude_background", action="store_true", default=True)
    p.add_argument("--max_n", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")

    # spectral graph + sequences
    p.add_argument("--spectral_mode", type=str, default="invariant", choices=["invariant","aligned","raw"])
    p.add_argument("--bands", type=int, default=12)
    p.add_argument("--k_reg", type=int, default=DEF_K_REG)
    p.add_argument("--grid_n", type=int, default=DEF_GRID_N)
    p.add_argument("--cell_thr", type=float, default=DEF_CELL_THR)
    p.add_argument("--pos_sigma", type=float, default=DEF_W_POS_SIGMA)
    p.add_argument("--int_sigma", type=float, default=DEF_W_INT_SIGMA)
    p.add_argument("--no_int_weight", action="store_true", help="disable intensity weighting in adjacency")

    # CV / general
    p.add_argument("--folds", type=int, default=DEF_FOLDS)
    p.add_argument("--seed", type=int, default=DEF_SEED)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--oversample", action="store_true", help="RandomOverSampler on training folds")

    # SNO
    p.add_argument("--sno_epochs", type=int, default=DEF_SNO_EPOCHS)
    p.add_argument("--sno_batch", type=int, default=DEF_SNO_BATCH)
    p.add_argument("--sno_rank", type=int, default=DEF_SNO_RANK)
    p.add_argument("--sno_hidden", type=int, default=DEF_SNO_HIDDEN)
    p.add_argument("--sno_dropout", type=float, default=DEF_SNO_DROPOUT)
    p.add_argument("--sno_lr", type=float, default=DEF_SNO_LR)
    p.add_argument("--sno_wd", type=float, default=DEF_SNO_WD)

    # LGBM
    p.add_argument("--lgbm_estimators", type=int, default=400)
    p.add_argument("--lgbm_lr", type=float, default=0.05)
    p.add_argument("--lgbm_max_depth", type=int, default=-1)
    p.add_argument("--lgbm_subsample", type=float, default=0.9)
    p.add_argument("--lgbm_colsample", type=float, default=0.8)
    p.add_argument("--lgbm_l2", type=float, default=0.0)
    p.add_argument("--lgbm_l1", type=float, default=0.0)

    # Fusion
    p.add_argument("--fusion_pca_dim", type=int, default=64)
    p.add_argument("--perm_all_folds", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    # labels
    if args.labels_json and os.path.isfile(args.labels_json):
        with open(args.labels_json, "r") as f:
            labels_map = json.load(f)
            # {name: [r,g,b]} or {name: (r,g,b)}
            labels_map = {k: tuple(v) for k,v in labels_map.items()}
    else:
        labels_map = DEFAULT_GBM_LABELS

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = Path(args.patch_dir) if args.patch_dir else (out_dir / "patches"); patch_dir.mkdir(parents=True, exist_ok=True)

    # build or load instances
    if args.build_patches:
        meta = build_or_load_gbm_instances(args.image_dir, args.mask_dir, str(patch_dir), labels_map, tol=2, min_contour_pts=40, force_rebuild=True)
    else:
        meta_path = patch_dir / "instances_meta.json"
        if not meta_path.exists():
            log.info("[warn] instances_meta.json missing; building now.")
            meta = build_or_load_gbm_instances(args.image_dir, args.mask_dir, str(patch_dir), labels_map, tol=2, min_contour_pts=40, force_rebuild=True)
        else:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            log.info(f"[GBM] loaded instances meta: {meta_path} (N={len(meta)})")

    # config bundle passed to feature builders
    cfg = dict(
        grid_n=args.grid_n,
        cell_thr=args.cell_thr,
        pos_sigma=args.pos_sigma,
        int_sigma=args.int_sigma,
        use_int_weight=(not args.no_int_weight),
        k_reg=args.k_reg
    )

    run_dataset(meta=meta, out_dir=out_dir, cfg=cfg, args=args)


if __name__ == "__main__":
    main()
