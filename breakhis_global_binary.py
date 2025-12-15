
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL BreakHis binary pipeline (five-result suite)

Built on SNO4 logic and helpers, with:
1) Baseline CNN→LGBM (image-only)
2) CNN⊕SNO→LGBM (late fusion with PCA + Hadamard interactions)
3) Spectral pooled (FULL) → LGBM (8ch region graph pooled stats + added features)
4) Spectral pooled (FULL) → LogisticRegression L2 (same features)
5) SNO→LGBM (use SNO pooled embedding as features for LGBM)

"FULL" pooled features stack extends SNO4 by adding:
- Fractal descriptors on spectral band energies (slope, Rényi Dq, lacunarity)
- Heat Kernel Signature (HKS) pooled over nodes
- Graph Spectral Wavelet energy (multi-scale heat kernels)
- Slepian concentration features (energy concentration in ROI)
- Persistent Homology summaries (Gudhi) on region centroids

Conserves: region graph construction, partial spectrum, spectral stabilization
(raw / aligned / invariant), patient-wise CV, strict padding, class balancing.

CLI example (per magnification):
python breakhis_global_binary.py \
  --breakhis_root /path/to/BreaKHis_v1/histology_slides/breast \
  --out_dir ./out_global \
  --magnifications 40X 100X 200X 400X \
  --spectral_mode invariant --bands 14 --k_reg 128 \
  --run_all

Requires: pygsp, numpy, pandas, scikit-learn, lightgbm (optional), torch,
torchvision (optional), gudhi (optional), opencv-python
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
log = logging.getLogger("global")
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix
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

def index_breakhis(root_or_list):
    MAGS = {"40X","100X","200X","400X"}
    EXT = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}
    roots = root_or_list if isinstance(root_or_list,(list,tuple)) else [root_or_list]
    rows=[]
    for root in roots:
        rp = Path(root)
        if not rp.exists(): raise FileNotFoundError(root)
        for magdir in (p for p in rp.rglob("*") if p.is_dir() and p.name.upper() in MAGS):
            parts = [q.lower() for q in magdir.parts]
            y = 0 if "benign" in parts else 1 if "malignant" in parts else None
            if y is None: continue
            subtype=""
            if "sob" in parts:
                try: subtype = parts[parts.index("sob")+1]
                except Exception: pass
            pid_folder = magdir.parent.name
            m = re.search(r"(\d{2}-\d+)$", pid_folder)
            pid = m.group(1) if m else pid_folder
            for f in magdir.iterdir():
                if f.is_file() and f.suffix.lower() in EXT:
                    rows.append({"path":str(f),"label":y,"patient_id":pid,"subtype":subtype,"magnification":magdir.name.upper()})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No images found under given roots.")
    return df.sort_values(["label","subtype","patient_id"]).reset_index(drop=True)

# ====================== features & graph ======================
def _rgb_to_he(rgb):
    """Ruifrok & Johnston H&E deconvolution."""
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
            cell = mask[y0:y1, x0:x1]
            frac = (cell>0).mean()
            if frac < cell_thr: continue
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
                mean_int = float(sub_gray.mean())
                mean_dst = float(sub_dist.mean())
                mean_sob = float(sub_sob.mean())
                var_lap  = float(sub_lap.var())
                ent_loc  = _local_entropy(sub_gray, np.ones_like(sub_gray, np.uint8)*255)
                mean_H   = float(np.maximum(sub_Hc, 0).mean())
                mean_E   = float(np.maximum(sub_Ec, 0).mean())
                mean_sat = float(sub_sat.mean())
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
    # z-score features per channel across nodes
    mu = F.mean(axis=0); sd = F.std(axis=0) + 1e-12
    Fz = (F - mu) / sd
    return G, Fz, W

# ====================== spectrum & stabilization ======================
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
    """Return slope and R^2 of log(y) ~ a*log(j) + b over j=1..K (y>=0)."""
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
    y = y - np.min(y) + eps  # ensure nonnegative
    K = int(y.size)
    vals = []
    for w in windows:
        if w > K:
            vals.append(1.0)
            continue
        cums = np.cumsum(np.r_[0.0, y])
        box = cums[w:] - cums[:-w]  # sliding window sums
        mu = float(np.mean(box))
        var = float(np.var(box))
        Λ = var / (mu*mu + eps) + 1.0
        vals.append(float(Λ))
    return vals

def fractal_encode_channels(E, eps=1e-12):
    """E: (C,K) band energies. Output: per-channel [slope,R2,D0.5,D1,D2,D4,Lac2,Lac4,Lac8] → 9*C dims"""
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
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)  # per-channel L2
    a_mat = (U.T @ X).T  # (C,K)
    anchors = _anchor_fields_for_graph(G, featsC)
    U_sf, sgn = _sign_fix_columns(U, anchors)
    a_mat *= sgn[None,:]
    return lam, U_sf, a_mat, anchors

# ====================== extra spectral features ======================
def hks_features(U: np.ndarray, lam: np.ndarray, t_scales: Tuple[float,...]=DEF_HKS_T,
                 pool: str = "meanstd") -> np.ndarray:
    """Compute HKS on nodes then pool across nodes at multiple scales.
    Returns concatenated vector (len(t_scales) * 2 if pool==meanstd else len(t_scales))."""
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
    """Simple Slepian-like concentration: choose ROI = top frac of first anchor (boundary-like),
    compute energy concentration of lowest n_vecs eigenvectors in ROI.
    Returns [mean, std, top-5 mean] of concentrations.
    """
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
    """Graph-heat wavelet energies per channel and scale. X: (N,C) node features (z-scored).
    Returns flattened (C * len(scales)) energies normalized per-channel.
    """
    if not HAS_PYGSP or G is None or X is None or X.size == 0:
        return np.zeros((X.shape[1]*len(t_scales),), np.float32) if X is not None else np.zeros(0, np.float32)
    L = G.L if isspmatrix(G.L) else csr_matrix(G.L)
    N = L.shape[0]
    C = X.shape[1]
    out = []
    # reuse a single eigensolver call per image if feasible
    k_eigs = min(max(10, int(0.8*N)), 256) if N > 20 else max(2, N-2)
    try:
        vals, vecs = eigsh(L, k=k_eigs, which='SM', tol=1e-3, maxiter=5000)
    except Exception:
        from numpy.linalg import eigh as dense_eigh
        Ld = L.toarray(); vals, vecs = dense_eigh(Ld)
    Ue = np.asarray(vecs); lam = np.asarray(vals)
    for c in range(C):
        f = X[:,c:c+1]  # (N,1)
        a = Ue.T @ f  # (k,1)
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
    """Simple PH summaries on region centroids using Vietoris–Rips.
    Returns [H0_count, H0_sum_life, H1_count, H1_sum_life, H1_persist_entropy].
    If Gudhi missing or graph too small, returns zeros.
    """
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

# ====================== sequences ======================
def sequences_for_path_pygsp(path, cfg, spectral_mode="invariant", n_bands=8):
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
        Kf = int(cfg["k_reg"])
        out = np.zeros((a_mat.shape[0], Kf), np.float32); n=min(Kf, a_mat.shape[1])
        out[:, :n] = a_mat[:, :n].astype(np.float32); return out
    if spectral_mode == "aligned":
        _, a_al = _band_align(U_sf, a_mat, lam, anchors, n_bands)
        Kf = int(cfg["k_reg"])
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

def compute_all_for_path(path, cfg, spectral_mode="invariant", n_bands=8):
    """Return (sequence CxK, full_features) for a path. Both are float32 arrays."""
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
        Kf = int(cfg["k_reg"]); out = np.zeros((a_mat.shape[0], Kf), np.float32); n=min(Kf, a_mat.shape[1]); out[:,:n]=a_mat[:,:n].astype(np.float32)
        Xseq = out
    elif spectral_mode == "aligned":
        _, a_al = _band_align(U_sf, a_mat, lam, anchors, n_bands)
        Kf = int(cfg["k_reg"]); out = np.zeros((a_al.shape[0], Kf), np.float32); n=min(Kf, a_al.shape[1]); out[:,:n]=a_al[:,:n].astype(np.float32)
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

def patient_stratified_splits(labels, patients, n_splits=DEF_FOLDS, seed=DEF_SEED):
    idx=np.arange(len(labels))
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(idx, labels, patients))
    uniq = np.array(sorted(list(set(patients))))
    maj=[]
    for u in uniq:
        labs = np.array(labels)[np.array(patients)==u]
        vals,cnts = np.unique(labs, return_counts=True)
        maj.append(vals[np.argmax(cnts)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits=[]
    for _, te_g in skf.split(uniq, maj):
        te_pat=set(uniq[te_g])
        te=[i for i,g in enumerate(patients) if g in te_pat]
        tr=[i for i in range(len(patients)) if i not in te]
        splits.append((np.array(tr), np.array(te)))
    return splits

def train_baselines(Xtr, ytr, Xte):
    if HAS_LGBM:
        clf_gbm=LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.03,
                               num_leaves=15, subsample=0.9, colsample_bytree=0.8,
                               reg_lambda=1.0, class_weight="balanced", feature_pre_filter=False,
                               random_state=DEF_SEED, n_jobs=-1)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf_gbm=GradientBoostingClassifier(n_estimators=600, learning_rate=0.05, max_depth=3, random_state=DEF_SEED)
    clf_gbm.fit(Xtr, ytr)
    prob_gbm = clf_gbm.predict_proba(Xte)[:,1] if hasattr(clf_gbm,"predict_proba") else clf_gbm.decision_function(Xte)
    clf_lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, class_weight="balanced", random_state=DEF_SEED)
    clf_lr.fit(Xtr, ytr); prob_lr = clf_lr.predict_proba(Xte)[:,1]
    return clf_gbm, prob_gbm, clf_lr, prob_lr

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
            # fallback to uninitialized weights if download blocked
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
    """Return 512-D embedding for the whole image via ResNet18 (global crop)."""
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

def load_or_build_cnn_embeddings(paths: np.ndarray, out_dir: Path, mag: str) -> np.ndarray:
    """Cache CNN embeddings aligned to pseq_all order for this magnification."""
    cache_path = Path(out_dir) / f"cnn_embed_{mag}.npz"
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            E = data["E"]; P = data["paths"]
            if len(P)==len(paths) and np.all(P==paths):
                log.info(f"[cache] loaded {cache_path}")
                return E.astype(np.float32)
        except Exception:
            pass
    # build fresh
    Emat = []
    it = tqdm(paths, desc=f"[CNNemb] {mag}")
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
        def __init__(self, C=8, K=128, hidden=64, rank=32, pdrop=0.1):
            super().__init__()
            self.mix1=SpectralMix(C,K,rank=rank, nonlin='softplus')
            self.mix2=SpectralMix(C,K,rank=rank, nonlin='softplus')
            self.drop=nn.Dropout(pdrop)
            self.pool=nn.AdaptiveAvgPool1d(1)
            self.head=nn.Sequential(nn.Flatten(), nn.Linear(C, hidden), nn.Softplus(), nn.Dropout(pdrop), nn.Linear(hidden,2))
        def forward(self, U):
            x=self.mix1(U); x=self.drop(x); x=self.mix2(x); z=self.pool(x).squeeze(-1); return self.head(z)
        # expose pooled features
        def forward_with_feat(self, U):
            x=self.mix1(U); x=self.drop(x); x=self.mix2(x); z=self.pool(x).squeeze(-1)
            logits=self.head(z); return logits, z

def train_eval_sno_cv(Xseq, y, splits, args):
    """Train SNO per fold; return metrics and per-fold pooled embeddings for LGBM + fusion."""
    if not HAS_TORCH or Xseq is None or len(Xseq)==0:
        log.info("PyTorch not available or no SNO sequences; skipping SNO."); return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N,C,K = Xseq.shape; log.info(f"SNO: N={N}, C={C}, K={K}, device={device}")
    sno_accs,sno_bals,sno_aucs=[],[],[]
    fold_feats=[]  # list of dicts with tr, te, Ztr, Zte, ytr, yte
    for fold,(tr,te) in enumerate(splits,1):
        Utr=torch.from_numpy(Xseq[tr]).to(device); Ute=torch.from_numpy(Xseq[te]).to(device)
        ytr_t=torch.from_numpy(y[tr].astype(np.int64)).to(device); yte_t=torch.from_numpy(y[te].astype(np.int64)).to(device)
        model=SNOClassifier(C=C, K=K, hidden=args.sno_hidden,
                            rank=min(args.sno_rank, K//2) if args.sno_rank>0 else None,
                            pdrop=args.sno_dropout).to(device)
        opt=torch.optim.AdamW(model.parameters(), lr=args.sno_lr, weight_decay=args.sno_wd)
        pos=(ytr_t==1).sum().item(); neg=(ytr_t==0).sum().item(); w1=(neg/max(1,pos)) if pos>0 else 1.0
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0,float(w1)], device=device))
        model.train(); idx_all=torch.arange(Utr.size(0), device=device)
        for _ in range(args.sno_epochs):
            idx=idx_all[torch.randperm(idx_all.numel())]
            for s in range(0,len(idx), args.sno_batch):
                b=idx[s:s+args.sno_batch]; logits=model(Utr[b]); loss=criterion(logits, ytr_t[b])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(Ute), dim=1)[:,1].detach().cpu().numpy()
            # pooled features for LGBM/fusion (train & test for this fold)
            _, z_tr = model.forward_with_feat(Utr)
            _, z_te = model.forward_with_feat(Ute)
            Ztr = z_tr.detach().cpu().numpy().astype(np.float32)
            Zte = z_te.detach().cpu().numpy().astype(np.float32)
        pred=(prob>=0.5).astype(int)
        acc=accuracy_score(y[te], pred); bal=balanced_accuracy_score(y[te], pred)
        try: auc=roc_auc_score(y[te], prob)
        except Exception: auc=float('nan')
        log.info(f"[SNO] Fold {fold}: Acc {acc:.3f} | BalAcc {bal:.3f} | AUC {auc:.3f}")
        sno_accs.append(acc); sno_bals.append(bal); sno_aucs.append(auc)
        fold_feats.append(dict(tr=tr, te=te, Ztr=Ztr, Zte=Zte, ytr=y[tr], yte=y[te]))
    log.info(f"\n*** SNO MEAN over folds ***")
    log.info(f"Accuracy: {np.mean(sno_accs):.3f} ± {np.std(sno_accs):.3f}")
    log.info(f"Balanced Acc: {np.mean(sno_bals):.3f} ± {np.std(sno_bals):.3f}")
    log.info(f"AUC: {np.nanmean(sno_aucs):.3f} ± {np.nanstd(sno_aucs):.3f}")
    return dict(acc=sno_accs, bal=sno_bals, auc=sno_aucs, folds=fold_feats)

# ====================== fusion utils ======================
def group_permutation_importance(clf, Xte, yte, groups: dict, seed=DEF_SEED):
    """Return dict with base_auc and per-group AUC drop after permutation."""
    rng = np.random.default_rng(seed)
    prob = clf.predict_proba(Xte)[:,1]
    try:
        base = roc_auc_score(yte, prob)
    except Exception:
        return {'base_auc': float('nan')}
    out = {'base_auc': float(base)}
    for name, idxs in groups.items():
        Xp = Xte.copy()
        for j in idxs:
            rng.shuffle(Xp[:, j])
        prob2 = clf.predict_proba(Xp)[:,1]
        try:
            auc2 = roc_auc_score(yte, prob2)
        except Exception:
            auc2 = float('nan')
        out[name] = float(base - auc2) if np.isfinite(auc2) else float('nan')
    return out

# ====================== run per magnification ======================
def run_mag(df_mag, out_dir, cfg, args):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"g{cfg['grid_n']}_cell{int(cfg['cell_thr']*100)}_K{cfg['k_reg']}_{args.spectral_mode}_B{args.bands}"
    cache_seq = Path(out_dir) / f"snoseq_{df_mag['magnification'].iloc[0]}_{tag}.npz"
    cache_full = Path(out_dir) / f"fullfeat_{df_mag['magnification'].iloc[0]}_{tag}.npz"

    Xseq_all=None; yseq_all=None; pseq_all=None; Xfull_all=None

    # ----- load or build caches -----
    load_ok = False
    if cache_seq.exists() and cache_full.exists() and not args.rebuild:
        try:
            d1=np.load(cache_seq, allow_pickle=False); d2=np.load(cache_full, allow_pickle=False)
            Xseq_all=d1["Xseq"]; yseq_all=d1["y"]; pseq_all=d1["paths"]
            Xfull_all=d2["Xfull"]
            if len(pseq_all)==len(Xfull_all):
                log.info(f"[cache] loaded {cache_seq} and {cache_full}")
                load_ok=True
        except Exception:
            load_ok=False

    if not load_ok:
        if not HAS_PYGSP: raise SystemExit("PyGSP not installed. pip install pygsp")
        Xseq_list=[]; Xfull_list=[]; y_list=[]; p_list=[]; miss=0
        it=tqdm(df_mag.iterrows(), total=len(df_mag), desc=f"[Build] {df_mag['magnification'].iloc[0]}")
        for _,r in it:
            try:
                Xseq, Xfull, rgb = compute_all_for_path(r["path"], cfg, spectral_mode=args.spectral_mode, n_bands=args.bands)
                if Xseq is None or not np.isfinite(Xseq).all(): miss+=1; continue
                if Xfull is None or not np.isfinite(Xfull).all(): miss+=1; continue
                Xseq_list.append(Xseq); Xfull_list.append(Xfull); y_list.append(int(r["label"])); p_list.append(r["path"])
            except Exception as e:
                miss+=1; log.info(f"[warn] extract fail: {r['path']} :: {e}")
        if not Xseq_list:
            log.info("No sequences extracted."); return
        # pad sequences to same K
        Kmax = max(a.shape[-1] for a in Xseq_list)
        fixed=[]
        for a in Xseq_list:
            C,K=a.shape
            out=np.zeros((C,Kmax), np.float32)
            out[:, :min(K,Kmax)] = a[:, :min(K,Kmax)]
            fixed.append(out)
        Xseq_all=np.stack(fixed, axis=0).astype(np.float32)
        Xfull_all=np.stack(Xfull_list, axis=0).astype(np.float32)
        yseq_all=np.array(y_list,np.int64); pseq_all=np.array(p_list)
        np.savez_compressed(cache_seq, Xseq=Xseq_all, y=yseq_all, paths=pseq_all)
        np.savez_compressed(cache_full, Xfull=Xfull_all, paths=pseq_all)
        log.info(f"[cache] saved {cache_seq} & {cache_full} | misses: {miss}")

    # ----- splits -----
    path2pid=dict(zip(df_mag['path'], df_mag['patient_id']))
    g=np.array([path2pid.get(p,"NA") for p in pseq_all])
    splits = patient_stratified_splits(yseq_all, g, n_splits=args.folds, seed=args.seed)

    # ----- (3) + (4): FULL pooled features baselines -----
    accs_gbm,bals_gbm,aucs_gbm=[],[],[]
    accs_lr, bals_lr, aucs_lr = [],[],[]
    for fold,(tr,te) in enumerate(splits,1):
        log.info(f"\n--- {df_mag['magnification'].iloc[0]} | Fold {fold}/{args.folds} ---  N_tr={len(tr)}  N_te={len(te)}")
        scaler=StandardScaler(); Xtr=scaler.fit_transform(Xfull_all[tr]); Xte=scaler.transform(Xfull_all[te])
        ytr=yseq_all[tr]; yte=yseq_all[te]
        if HAS_IMB and args.oversample:
            try:
                ros=RandomOverSampler(sampling_strategy="auto", random_state=args.seed)
                Xtr,ytr=ros.fit_resample(Xtr,ytr)
            except Exception as e:
                log.info("[warn] oversampler failed: %s", e)
        Xtr,Xte,_=sanitize_X(Xtr,Xte)
        clf_gbm,prob_gbm,clf_lr,prob_lr = train_baselines(Xtr,ytr,Xte)

        pred=(prob_gbm>=0.5).astype(int); acc=accuracy_score(yte,pred); bal=balanced_accuracy_score(yte,pred)
        try: auc=roc_auc_score(yte,prob_gbm)
        except Exception: auc=float('nan')
        log.info(f"[3: FULL→LGBM] Acc: {acc:.3f} | BalAcc: {bal:.3f} | AUC: {auc:.3f}")
        accs_gbm.append(acc); bals_gbm.append(bal); aucs_gbm.append(auc)

        pred=(prob_lr>=0.5).astype(int); acc=accuracy_score(yte,pred); bal=balanced_accuracy_score(yte,pred)
        try: auc=roc_auc_score(yte,prob_lr)
        except Exception: auc=float('nan')
        log.info(f"[4: FULL→LR-L2] Acc: {acc:.3f} | BalAcc: {bal:.3f} | AUC: {auc:.3f}")
        accs_lr.append(acc); bals_lr.append(bal); aucs_lr.append(auc)

        log.info("Confusion Matrix (FULL→LGBM):\n%s", confusion_matrix(yte, (prob_gbm>=0.5).astype(int), labels=[0,1]))
        log.info(classification_report(yte, (prob_gbm>=0.5).astype(int), labels=[0,1], target_names=["benign","malignant"], zero_division=0))

    def summarize(name, accs, bals, aucs):
        accs=np.array(accs); bals=np.array(bals); aucs=np.array(aucs)
        log.info(f"\n*** {name} MEAN over folds ***")
        log.info(f"Accuracy: {accs.mean():.3f} ± {accs.std():.3f}")
        log.info(f"Balanced Acc: {bals.mean():.3f} ± {bals.std():.3f}")
        log.info(f"AUC: {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")

    summarize("3: FULL→LGBM", accs_gbm, bals_gbm, aucs_gbm)
    summarize("4: FULL→LR-L2", accs_lr,  bals_lr,  aucs_lr)

    # ----- sequences used for SNO -----
    sno_out = train_eval_sno_cv(Xseq_all.astype(np.float32), yseq_all.astype(np.int64), splits, args) if HAS_TORCH else None

    # ----- (5): SNO→LGBM (pooled SNO embedding as features) -----
    if sno_out is not None and 'folds' in sno_out:
        acc_sno_lgbm, bal_sno_lgbm, auc_sno_lgbm = [], [], []
        for fold_id, ((tr,te), fd) in enumerate(zip(splits, sno_out['folds']), 1):
            Ztr, Zte = fd['Ztr'], fd['Zte']
            ytr, yte = fd['ytr'], fd['yte']
            scalerZ = StandardScaler().fit(Ztr)
            XtrZ = scalerZ.transform(Ztr); XteZ = scalerZ.transform(Zte)
            if HAS_LGBM:
                clf = LGBMClassifier(
                    objective="binary", n_estimators=800, learning_rate=0.05,
                    num_leaves=31, subsample=0.9, colsample_bytree=0.8,
                    reg_lambda=1.0, class_weight="balanced", feature_pre_filter=False,
                    random_state=args.seed, n_jobs=-1
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                clf=GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=args.seed)
            clf.fit(XtrZ, ytr)
            prob = clf.predict_proba(XteZ)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(XteZ)
            pred = (prob >= 0.5).astype(int)
            acc = accuracy_score(yte, pred); bal = balanced_accuracy_score(yte, pred)
            try: auc = roc_auc_score(yte, prob)
            except Exception: auc = float('nan')
            log.info(f"[5: SNO→LGBM] Fold {fold_id}: Acc {acc:.3f} | BalAcc {bal:.3f} | AUC {auc:.3f}")
            acc_sno_lgbm.append(acc); bal_sno_lgbm.append(bal); auc_sno_lgbm.append(auc)
        summarize("5: SNO→LGBM", acc_sno_lgbm, bal_sno_lgbm, auc_sno_lgbm)
    else:
        log.info("SNO not available; skipping 5: SNO→LGBM.")

    # ----- CNN embeddings (for baseline + fusion) -----
    if not HAS_TORCHVISION:
        log.info("torchvision not available; skipping 1: CNN→LGBM and 2: CNN⊕SNO→LGBM.")
        return

    Emat = load_or_build_cnn_embeddings(pseq_all, out_dir, df_mag['magnification'].iloc[0])  # (N,512)

    # ----- (1): Baseline CNN→LGBM -----
    acc_cnn, bal_cnn, auc_cnn = [], [], []
    for fold,(tr,te) in enumerate(splits,1):
        Etr = Emat[tr]; Ete = Emat[te]; ytr = yseq_all[tr]; yte = yseq_all[te]
        scaler = StandardScaler().fit(Etr)
        Xtr = scaler.transform(Etr); Xte = scaler.transform(Ete)
        if HAS_LGBM:
            clf = LGBMClassifier(objective="binary", n_estimators=1200, learning_rate=0.03,
                                 num_leaves=31, subsample=0.9, colsample_bytree=0.8,
                                 reg_lambda=1.0, class_weight="balanced", feature_pre_filter=False,
                                 random_state=args.seed, n_jobs=-1)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf=GradientBoostingClassifier(n_estimators=800, learning_rate=0.05, max_depth=3, random_state=args.seed)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(Xte)
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(yte, pred); bal = balanced_accuracy_score(yte, pred)
        try: auc = roc_auc_score(yte, prob)
        except Exception: auc = float('nan')
        log.info(f"[1: CNN→LGBM] Fold {fold}: Acc {acc:.3f} | BalAcc {bal:.3f} | AUC {auc:.3f}")
        acc_cnn.append(acc); bal_cnn.append(bal); auc_cnn.append(auc)
    summarize("1: CNN→LGBM", acc_cnn, bal_cnn, auc_cnn)

    # ----- (2): CNN⊕SNO→LGBM (late fusion) -----
    if sno_out is None or 'folds' not in sno_out:
        log.info("No SNO output available; skipping fusion.")
        return
    acc_fused, bal_fused, auc_fused = [], [], []
    for fold_id, ((tr,te), fd) in enumerate(zip(splits, sno_out['folds']), 1):
        # sanity: indices match
        assert np.all(fd['tr']==tr) and np.all(fd['te']==te), "Fold indices mismatch."
        Ztr, Zte = fd['Ztr'], fd['Zte']                   # (n_tr, C), (n_te, C)
        ytr, yte = fd['ytr'], fd['yte']
        Cdim = Ztr.shape[1]                               # match dimension for PCA
        # PCA on training CNN only -> Cdim
        pca = PCA(n_components=Cdim, svd_solver="auto", random_state=args.seed)
        Etr = pca.fit_transform(Emat[tr]); Ete = pca.transform(Emat[te])
        # standardize groups separately
        scaler_spec = StandardScaler().fit(Ztr)
        scaler_cnn  = StandardScaler().fit(Etr)
        Ztr_s = scaler_spec.transform(Ztr); Zte_s = scaler_spec.transform(Zte)
        Ctr_s = scaler_cnn.transform(Etr); Cte_s = scaler_cnn.transform(Ete)
        # interaction: Hadamard (elementwise) product
        Itr = Ztr_s * Ctr_s; Ite = Zte_s * Cte_s
        # concatenate blocks
        Xtr = np.hstack([Ztr_s, Ctr_s, Itr]); Xte = np.hstack([Zte_s, Cte_s, Ite])
        if HAS_LGBM:
            clf = LGBMClassifier(
                objective="binary", n_estimators=1200, learning_rate=0.03,
                num_leaves=31, subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, class_weight="balanced", feature_pre_filter=False,
                random_state=args.seed, n_jobs=-1
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf=GradientBoostingClassifier(n_estimators=800, learning_rate=0.05, max_depth=3, random_state=args.seed)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(yte, pred); bal = balanced_accuracy_score(yte, pred)
        try: auc = roc_auc_score(yte, prob)
        except Exception: auc = float("nan")
        # group permutation importance (optional, printed)
        D = Cdim
        groups = {'SNO': list(range(0, D)),
                  'CNNpca': list(range(D, 2*D)),
                  'INT': list(range(2*D, 3*D))}
        imp = group_permutation_importance(clf, Xte, yte, groups, seed=args.seed)
        log.info(f"[2: FUSED SNO⊕CNN→LGBM] Fold {fold_id}: Acc {acc:.3f} | BalAcc {bal:.3f} | AUC {auc:.3f}")
        log.info(f"[FUSED IMPORTANCE] AUC drop: {imp}")
        acc_fused.append(acc); bal_fused.append(bal); auc_fused.append(auc)
    summarize("2: FUSED (SNO⊕CNN→LGBM)", acc_fused, bal_fused, auc_fused)

# ====================== cli ======================
def parse_args():
    p = argparse.ArgumentParser(description="Global BreakHis binary: SNO4 + fractal/HKS/wavelet/Slepian/persistent + CNN/SNO fusion (5-result suite)")
    p.add_argument("--breakhis_root", type=str, required=True, help="Path to .../BreaKHis_v1/histology_slides/breast")
    p.add_argument("--out_dir", type=str, default="./global_out")
    p.add_argument("--folds", type=int, default=DEF_FOLDS)
    p.add_argument("--seed", type=int, default=DEF_SEED)
    p.add_argument("--magnifications", type=str, nargs="*", default=["40X","100X","200X","400X"])

    # graph / spectral
    p.add_argument("--grid_n", type=int, default=DEF_GRID_N)
    p.add_argument("--cell_thr", type=float, default=DEF_CELL_THR)
    p.add_argument("--pos_sigma", type=float, default=DEF_W_POS_SIGMA)
    p.add_argument("--int_sigma", type=float, default=DEF_W_INT_SIGMA)
    p.add_argument("--no_int_weight", action="store_true")
    p.add_argument("--k_reg", type=int, default=DEF_K_REG)

    # spectral stabilization
    p.add_argument("--spectral_mode", type=str, default="invariant", choices=["raw","aligned","invariant"])
    p.add_argument("--bands", type=int, default=14)

    # baselines / cache
    p.add_argument("--oversample", action="store_true")
    p.add_argument("--rebuild", action="store_true")

    # SNO hparams
    p.add_argument("--sno_epochs", type=int, default=DEF_SNO_EPOCHS)
    p.add_argument("--sno_batch", type=int, default=DEF_SNO_BATCH)
    p.add_argument("--sno_rank", type=int, default=DEF_SNO_RANK)
    p.add_argument("--sno_hidden", type=int, default=DEF_SNO_HIDDEN)
    p.add_argument("--sno_dropout", type=float, default=DEF_SNO_DROPOUT)
    p.add_argument("--sno_lr", type=float, default=DEF_SNO_LR)
    p.add_argument("--sno_wd", type=float, default=DEF_SNO_WD)

    # suite control
    p.add_argument("--run_all", action="store_true", help="run all mags; otherwise only the ones present with >=50 imgs")
    return p.parse_args()

def main():
    args = parse_args()
    if not HAS_PYGSP: raise SystemExit("PyGSP not installed. pip install pygsp")
    set_global_seed(args.seed)
    cfg = dict(
        grid_n=args.grid_n,
        cell_thr=args.cell_thr,
        pos_sigma=args.pos_sigma,
        int_sigma=args.int_sigma,
        use_int_weight=(not args.no_int_weight),
        k_reg=args.k_reg,
    )
    log.info("Config:\n%s", json.dumps(cfg, indent=2))
    df = index_breakhis(args.breakhis_root)
    log.info("Total images indexed: %d", len(df))
    mags = [m for m in args.magnifications if (args.run_all or (df.magnification==m).sum()>=50)]
    for mag in mags:
        dfm = df[df.magnification==mag].copy()
        if len(dfm) < 50:
            log.info("\nSkipping %s (too few images: %d)", mag, len(dfm)); continue
        log.info(f"\n===== {mag} | N={len(dfm)} =====")
        run_mag(dfm, out_dir=args.out_dir, cfg=cfg, args=args)

if __name__ == "__main__":
    main()
