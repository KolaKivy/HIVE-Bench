from __future__ import annotations
import argparse, os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"Analyze/outputs/tsne"; DEST=OUT/"split_half"

def run(dataset,name,args):
 from sklearn.decomposition import PCA
 from sklearn.manifold import TSNE
 import matplotlib.pyplot as plt
 cache=OUT/"cache"/dataset/name; files=sorted(cache.glob("shard_*.pt"))
 if not files: raise FileNotFoundError(cache)
 maxima={}
 for f in tqdm(files,desc=f"scan episodes {dataset}:{name}"):
  d=torch.load(f,map_location="cpu",weights_only=False); t=np.asarray(d["task"]); e=np.asarray(d["episode"])
  for k in np.unique(t):
   q=e[t==k]
   if len(q): maxima[int(k)]=max(maxima.get(int(k),-1),int(q.max()))
 rng=np.random.default_rng(args.seed); reservoirs={(h,k):[] for h in (0,1) for k in maxima}
 for f in tqdm(files,desc=f"sample split {dataset}:{name}"):
  d=torch.load(f,map_location="cpu",weights_only=False); z=d["tokens"].float().numpy(); t=np.asarray(d["task"]); e=np.asarray(d["episode"])
  for k in np.unique(t):
   for h in (0,1):
    mid=(maxima[int(k)]+1)//2; m=(t==k)&((e<mid) if h==0 else (e>=mid))
    if not m.any(): continue
    arr=z[m].reshape(-1,z.shape[-1]); cur=reservoirs[(h,int(k))]; cur.extend(arr)
    if len(cur)>args.points_per_task: reservoirs[(h,int(k))]=[cur[i] for i in rng.choice(len(cur),args.points_per_task,replace=False)]
 xs=[]; labs=[]
 for h in (0,1):
  for k in maxima:
   a=reservoirs[(h,k)]
   if a: xs.append(np.asarray(a)); labs.extend([h]*len(a))
 x=np.concatenate(xs); y=np.asarray(labs); x=PCA(n_components=min(args.pca_dim,x.shape[1],len(x)-1),random_state=args.seed).fit_transform(x)
 perp=min(args.perplexity,max(5,(len(x)-1)//3)); emb=TSNE(n_components=2,perplexity=perp,init="pca",learning_rate="auto",random_state=args.seed,max_iter=args.max_iter).fit_transform(x)
 fig,ax=plt.subplots(figsize=(15,10)); colors=["#c62828","#1565c0"]
 for h,label in enumerate(("first half","second half")):
  m=y==h; ax.scatter(emb[m,0],emb[m,1],s=args.point_size,alpha=args.alpha,c=colors[h],linewidths=0,rasterized=True,label=label)
 ax.set_title(f"{name} - {dataset} split-half t-SNE",fontsize=18,fontweight="bold"); ax.set_xlabel("t-SNE 1",fontsize=16); ax.set_ylabel("t-SNE 2",fontsize=16); ax.tick_params(labelsize=13); ax.legend(title="Episode subset",fontsize=11,title_fontsize=12,frameon=True); fig.tight_layout()
 out=DEST/"figures"; out.mkdir(parents=True,exist_ok=True); stem=f"{dataset}_{name}_split_half"; fig.savefig(out/(stem+".png"),dpi=300); fig.savefig(out/(stem+".pdf")); plt.close(fig); print(f"saved {out/(stem+'.png')} ({len(x)} points)",flush=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True,choices=["robotwin","robocasa"]); ap.add_argument("--name",required=True); ap.add_argument("--points-per-task",type=int,default=5000); ap.add_argument("--pca-dim",type=int,default=50); ap.add_argument("--perplexity",type=float,default=30); ap.add_argument("--max-iter",type=int,default=1000); ap.add_argument("--point-size",type=float,default=3); ap.add_argument("--alpha",type=float,default=.8); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args(); run(a.dataset,a.name,a)
if __name__=="__main__": main()
