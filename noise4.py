import os, argparse, math, numpy as np, pandas as pd, cv2, torch
from scipy import sparse
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg
from PIL import Image
import scanpy as sc
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _find_first(p):
    import glob
    r=sorted(glob.glob(p))
    return r[0] if r else None

def load_visium_h5(dataset_dir,sample_id="",use_image="hires"):
    if sample_id:
        sd=os.path.join(dataset_dir,sample_id)
    else:
        subs=[d for d in sorted(os.listdir(dataset_dir)) if d.startswith("GSM")]
        sd=os.path.join(dataset_dir,subs[0])
    h5=_find_first(os.path.join(sd,"*filtered_feature_bc_matrix.h5"))
    ad=sc.read_10x_h5(h5)
    ad.var_names_make_unique()
    sp=os.path.join(sd,"spatial")
    hires=_find_first(os.path.join(sp,"*tissue_hires_image.png"))
    poscsv=_find_first(os.path.join(sp,"*tissue_positions_list.csv"))
    img=Image.open(hires).convert("RGB")
    pos=pd.read_csv(poscsv,header=None)
    pos.columns=["barcode","in_tissue","array_row","array_col","px","py"]
    pos=pos[pos["barcode"].isin(ad.obs_names)].copy()
    pos=pos.set_index("barcode").loc[ad.obs_names]
    coords=pos[["px","py"]].values.astype(np.float32)
    if sparse.issparse(ad.X): ad.X=ad.X.toarray().astype(np.float32)
    return ad,np.array(img),coords,os.path.basename(sd)

def spot_patch_features(img,coords,patch_px=64):
    h,w,_=img.shape
    half=patch_px//2
    feats=[]
    for (x,y) in coords.astype(int):
        x0,x1=max(0,x-half),min(w,x+half)
        y0,y1=max(0,y-half),min(h,y+half)
        if x1-x0<8 or y1-y0<8:
            feats.append(np.zeros(8,np.float32));continue
        patch=img[y0:y1,x0:x1,:]
        gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
        sobx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        soby=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        sob_mag=np.sqrt(sobx*sobx+soby*soby)
        lap=cv2.Laplacian(gray,cv2.CV_64F,ksize=3)
        hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
        f=np.array([gray.mean(),sob_mag.mean(),lap.var(),gray.var(),gray.std(),hsv[...,1].mean(),gray.max(),gray.min()],np.float32)
        feats.append(f)
    return np.vstack(feats).astype(np.float32)

def poisson_virtual_spots(coords,factor=2.0,min_dist=None,seed=42,max_trials=200000):
    rng=np.random.default_rng(seed)
    N=coords.shape[0]
    target=int(max(N,int(N*factor)))
    cx,cy=coords[:,0],coords[:,1]
    minx,maxx=float(cx.min()),float(cx.max())
    miny,maxy=float(cy.min()),float(cy.max())
    if min_dist is None:
        tree=cKDTree(coords)
        d,_=tree.query(coords,k=2)
        nn1=d[:,1]
        min_dist=float(np.median(nn1[np.isfinite(nn1)]))
    pts=[]
    base_tree=cKDTree(coords)
    trials=0
    while len(pts)<target-N and trials<max_trials:
        trials+=1
        x=float(rng.uniform(minx,maxx))
        y=float(rng.uniform(miny,maxy))
        d0=base_tree.query([(x,y)],k=1)[0]
        d0=float(d0 if np.isscalar(d0) else d0[0])
        if d0 < min_dist: continue
        if len(pts)>0:
            arr=np.asarray(pts,dtype=np.float32)
            if arr.ndim==1: arr=arr.reshape(1,2)
            d1=cKDTree(arr).query([(x,y)],k=1)[0]
            d1=float(d1 if np.isscalar(d1) else d1[0])
            if d1 < min_dist: continue
        pts.append((x,y))
    return np.array(pts,np.float32)

def build_guided_graph(coords, feats, sigma=50.0, eps=0.0, max_nn=24, tau="auto", guided=True):
    coords = np.asarray(coords, dtype=np.float32)
    feats  = np.asarray(feats,  dtype=np.float32)
    N = coords.shape[0]
    tree = cKDTree(coords)
    k = int(min(max_nn, N))
    dists, idxs = tree.query(coords, k=k)
    if np.isscalar(dists):
        dists = dists.reshape(N, 1)
        idxs  = idxs.reshape(N, 1)

    if eps == 0.0 or eps is None:
        base = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
        eps = 1.5 * float(np.median(base[np.isfinite(base)]))

    if isinstance(sigma, str):
        sigma = float(sigma)
    sigma2 = max(float(sigma) * float(sigma), 1e-12)

    if isinstance(tau, str) and tau.strip().lower() != "auto":
        tau = float(tau)

    if (isinstance(tau, str) and tau.strip().lower() == "auto"):
        m = min(200, N)
        rng = np.random.default_rng(0)
        a = rng.choice(N, size=m, replace=False)
        vals = []
        for i in range(m):
            fi = feats[a[i]]
            for j in range(i + 1, m):
                vals.append(float(np.linalg.norm(fi - feats[a[j]])))
        tau = float(np.median(vals) + 1e-12) if vals else 1.0
    tau2 = max(float(tau) * float(tau), 1e-12)

    rows, cols, wts = [], [], []
    for i in range(N):
        di = dists[i]; ji = idxs[i]
        for d, j in zip(di, ji):
            if j == i or (not np.isfinite(d)) or d > eps:
                continue
            w = math.exp(-(d * d) / sigma2)
            if guided:
                diff = feats[i] - feats[j]
                w *= math.exp(-(float(np.dot(diff, diff))) / tau2)
            rows.append(i); cols.append(j); wts.append(w)

    rows2 = rows + cols
    cols2 = cols + rows
    vals2 = wts  + wts
    W = coo_matrix((vals2, (rows2, cols2)), shape=(N, N)).tocsr()
    L = csgraph.laplacian(W, normed=True)
    return W, L


def cg_solve(A,b,tol=1e-5,maxit=200):
    x,info=cg(A,b,maxiter=maxit,rtol=tol,atol=0.0)
    if isinstance(x,np.ndarray):
        return x.astype(np.float32)
    return np.asarray(x,dtype=np.float32)

def denoise_gene(y,is_lab,L,alpha=0.1,beta=0.0,g_prior=None):
    N=L.shape[0]
    M=np.zeros(N,np.float32)
    M[is_lab]=1.0
    A=L*alpha+sparse.diags(M+beta)
    b=M*y+(beta*g_prior if g_prior is not None else M*y)
    return cg_solve(A,b)

def rasterize(coords,vals,img_shape,rad=8):
    h,w,_=img_shape
    canvas=np.zeros((h,w),np.float32)
    counts=np.zeros((h,w),np.float32)
    xy=coords.astype(int)
    r=int(rad)
    for (x,y),v in zip(xy,vals):
        x0,x1=max(0,x-r),min(w,x+r)
        y0,y1=max(0,y-r),min(h,y+r)
        canvas[y0:y1,x0:x1]+=v
        counts[y0:y1,x0:x1]+=1.0
    counts[counts==0]=1.0
    return canvas/counts

def morans_I(vals,W):
    N=W.shape[0]
    wsum=W.sum()
    z=vals-vals.mean()
    num=float(z@(W@z))
    den=float((z*z).sum())
    return (N/wsum)*num/den if den>0 and wsum>0 else 0.0

def knn_xy_predict(train_idx,test_idx,coords,y,k=8):
    tr=coords[train_idx]
    te=coords[test_idx]
    tree=cKDTree(tr)
    kk=min(k,len(train_idx)) if len(train_idx)>0 else 1
    d,nn=tree.query(te,k=kk)
    if kk==1:
        d=d.reshape(-1,1); nn=nn.reshape(-1,1)
    w=1.0/(d+1e-6)
    s=np.sum(w,axis=1,keepdims=True)
    s[s==0]=1.0
    w=w/s
    pred=np.sum(y[train_idx][nn]*w[...,None],axis=1)
    return np.nan_to_num(pred,copy=False)

def rbf_xy_predict(train_idx,test_idx,coords,y,sigma=50.0):
    tr=coords[train_idx]
    te=coords[test_idx]
    tree=cKDTree(tr)
    k=min(24,len(train_idx)) if len(train_idx)>0 else 1
    d,nn=tree.query(te,k=k)
    if k==1:
        d=d.reshape(-1,1); nn=nn.reshape(-1,1)
    w=np.exp(-(d*d)/max(1e-6,sigma*sigma))
    s=np.sum(w,axis=1,keepdims=True)
    s[s==0]=1.0
    w=w/s
    pred=np.sum(y[train_idx][nn]*w[...,None],axis=1)
    return np.nan_to_num(pred,copy=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--sample_id",default="")
    ap.add_argument("--patch_px",type=int,default=64)
    ap.add_argument("--genes",type=int,default=20)
    ap.add_argument("--ssr_factor",type=float,default=2.0)
    ap.add_argument("--ssr_alpha",type=float,default=0.1)
    ap.add_argument("--ssr_beta",type=float,default=0.0)
    ap.add_argument("--ssr_guided",type=int,default=1)
    ap.add_argument("--ssr_sigma",type=float,default=50.0)
    ap.add_argument("--ssr_eps",type=float,default=0.0)
    ap.add_argument("--ssr_max_nn",type=int,default=24)
    ap.add_argument("--ssr_tau",default="auto")
    ap.add_argument("--ssr_eval_frac",type=float,default=0.5)
    ap.add_argument("--downsample_frac",type=float,default=1.0)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--min_expr_frac",type=float,default=0.05)
    ap.add_argument("--min_std",type=float,default=1e-3)
    ap.add_argument("--min_moran",type=float,default=0.0)
    args=ap.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    adata,img,coords,sample_name=load_visium_h5(args.dataset_dir,args.sample_id)
    genes=list(adata.var_names[:args.genes])
    Y=adata[:,genes].X.astype(np.float32)
    Y_log=np.log1p(np.maximum(Y,0))
    X_base=spot_patch_features(img,coords,patch_px=args.patch_px)
    virt=poisson_virtual_spots(coords,factor=args.ssr_factor,min_dist=None,seed=42)
    Xv=spot_patch_features(img,virt,patch_px=args.patch_px)
    coords_all=np.vstack([coords,virt])
    X_all=np.vstack([X_base,Xv])
    W,L=build_guided_graph(coords_all,X_all,sigma=args.ssr_sigma,eps=args.ssr_eps,max_nn=args.ssr_max_nn,tau=args.ssr_tau,guided=bool(args.ssr_guided))
    rng=np.random.default_rng(args.seed)
    N = coords.shape[0]
    if args.downsample_frac < 1.0:
        keep = max(1, int(round(N * args.downsample_frac)))
        idx = np.arange(N)
        train_idx = np.sort(rng.choice(idx, size=keep, replace=False))
        test_idx = np.array([i for i in idx if i not in set(train_idx)], dtype=int)
    else:
        msk=np.zeros(N,bool)
        keep=int(max(1,round(N*(1.0-args.ssr_eval_frac))))
        step=max(1,N//keep)
        sel=np.arange(0,N,step)[:keep]
        msk[sel]=True
        train_idx=np.where(msk)[0]
        test_idx=np.where(~msk)[0]
    is_lab=np.zeros(coords_all.shape[0],bool)
    is_lab[train_idx]=True
    preds=[]
    for j in range(Y_log.shape[1]):
        y=np.zeros(coords_all.shape[0],np.float32)
        y[train_idx]=Y_log[train_idx,j]
        f=denoise_gene(y,is_lab,L,alpha=args.ssr_alpha,beta=args.ssr_beta,g_prior=None)
        preds.append(f)
    Y_all=np.stack(preds,axis=1)
    np.save(os.path.join(args.out_dir,"Y_log_ssr.npy"),Y_all)
    np.save(os.path.join(args.out_dir,"Y_log_ssr_virtual.npy"),Y_all[coords.shape[0]:])
    pd.DataFrame(virt,columns=["x","y"]).to_csv(os.path.join(args.out_dir,"virtual_coords.csv"),index=False)
    Y_true=Y_log[test_idx]
    Y_pred_ssr=Y_all[test_idx]
    pd.DataFrame({"slide":[sample_name],"N_total":[N],"N_train":[len(train_idx)],"N_test":[len(test_idx)],"downsample_frac":[args.downsample_frac]}).to_csv(os.path.join(args.out_dir,"split_info.csv"),index=False)
    W_test=W[test_idx][:,test_idx]
    rows=[]
    def compute_metrics(yt,yp,coords_sub,genename,method):
        expr_frac=float((yt>0).mean())
        sd=float(yt.std())
        mor=float(morans_I(yt,W_test)) if W_test.nnz>0 else 0.0
        if not (expr_frac>=args.min_expr_frac and sd>=args.min_std and mor>=args.min_moran):
            return None
        pcc=float(np.corrcoef(yt,yp)[0,1]) if yt.std()>0 and yp.std()>0 else 0.0
        rmse=float(np.sqrt(mean_squared_error(yt,yp)))
        mae=float(mean_absolute_error(yt,yp))
        r2=float(r2_score(yt,yp)) if yt.std()>0 else 0.0
        gt=rasterize(coords_sub,yt,img.shape,rad=8)
        pr=rasterize(coords_sub,yp,img.shape,rad=8)
        dr=max(1e-6,float(gt.max()-gt.min()))
        s=float(ssim(gt,pr,data_range=dr))
        p=float(psnr(gt,pr,data_range=dr))
        return {"gene":genename,"method":method,"PCC":pcc,"RMSE":rmse,"MAE":mae,"R2":r2,"SSIM":s,"PSNR":p}
    for j,g in enumerate(genes):
        m=compute_metrics(Y_true[:,j],Y_pred_ssr[:,j],coords[test_idx],g,"ssr")
        if m is not None: rows.append(m)
    Y_knn=knn_xy_predict(train_idx,test_idx,coords,Y_log,k=8)
    for j,g in enumerate(genes):
        m=compute_metrics(Y_true[:,j],Y_knn[:,j],coords[test_idx],g,"knn_xy")
        if m is not None: rows.append(m)
    Y_rbf=rbf_xy_predict(train_idx,test_idx,coords,Y_log,sigma=args.ssr_sigma)
    for j,g in enumerate(genes):
        m=compute_metrics(Y_true[:,j],Y_rbf[:,j],coords[test_idx],g,"rbf_xy")
        if m is not None: rows.append(m)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir,"metrics_ssr.csv"),index=False)
    os.makedirs(os.path.join(args.out_dir,"ssr_maps"),exist_ok=True)
    gshow=min(8,len(genes))
    shown=0
    j=0
    while shown<gshow and j<len(genes):
        yt=Y_true[:,j]
        sd=float(yt.std())
        if sd>=args.min_std:
            gt=rasterize(coords[test_idx],yt,img.shape,rad=8)
            pr=rasterize(coords[test_idx],Y_pred_ssr[:,j],img.shape,rad=8)
            cv2.imwrite(os.path.join(args.out_dir,"ssr_maps",f"{genes[j]}_gt.png"),(gt-gt.min())/(gt.max()-gt.min()+1e-6)*255)
            cv2.imwrite(os.path.join(args.out_dir,"ssr_maps",f"{genes[j]}_ssr.png"),(pr-pr.min())/(pr.max()-pr.min()+1e-6)*255)
            shown+=1
        j+=1

if __name__=="__main__":
    main()
