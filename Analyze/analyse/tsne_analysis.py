"""Extract and visualize vision-encoder patch-token representations.

Examples (run from HIVE-Bench):
  python -m Analyze.analyse.tsne_analysis extract --dataset robotwin --encoder clip
  python -m Analyze.analyse.tsne_analysis plot --dataset robotwin --encoder clip
  python -m Analyze.analyse.tsne_analysis all --dataset both
"""
from __future__ import annotations
import argparse, json, os, sys
# Limit BLAS/OpenMP thread creation before importing NumPy/sklearn. Without
# this, large PCA runs can exceed OpenBLAS metadata limits on high-core hosts.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = ROOT / "playground/RoboTwin_LeRobot/Randomized"
ROBOCASA_ROOT = ROOT / "playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
ROBOTWIN_TASKS = ["adjust_bottle","beat_block_hammer","click_alarmclock","handover_block","lift_pot","move_playingcard_away","open_laptop","place_burger_fries","place_can_basket","rotate_qrcode","stamp_seal","turn_switch"]
ROBOCASA_TASKS = ["gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"]
ENCODERS = ["dinov2_base","dinov3_base","spa_base","vc1_base","clip","siglip","siglip2","mae","vit","radio","cradio","theia","internvit","voltron","vggt_omega","vjepa2_base","lingbot_large"]
OUT = ROOT / "Analyze/outputs/tsne"

def dataset_spec(name):
    if name == "robotwin": return ROBOTWIN_ROOT, ROBOTWIN_TASKS, "observation.images.cam_high", 500
    if name == "robocasa": return ROBOCASA_ROOT, ROBOCASA_TASKS, "observation.images.ego_view", 1000
    raise ValueError(name)

def video_files(task_dir, camera, limit):
    base = task_dir / "videos" / "chunk-000" / camera
    files = sorted(base.glob("*.mp4")) if base.exists() else []
    if not files:
        # camera names can be encoded as directory suffixes
        files = sorted(task_dir.glob(f"videos/chunk-*/{camera}/*.mp4"))
    return files[:limit]

def build_encoder(name, device, checkpoint=None):
    sys.path.insert(0, str(ROOT / "Policy"))
    from hivebench.model.framework.DinoGR00T import _build_vision_encoder
    model = _build_vision_encoder(name).eval().to(device)
    if checkpoint:
        from Analyze.analyse.idm_fdm_model import load_finetuned_vision_encoder
        load_finetuned_vision_encoder(model, checkpoint)
    for p in model.parameters(): p.requires_grad_(False)
    return model

@torch.inference_mode()
def extract(args):
    root, tasks, camera, limit = dataset_spec(args.dataset)
    cache_name = args.name or (args.encoder + "_ft" if args.checkpoint else args.encoder)
    out = OUT / "cache" / args.dataset / cache_name; out.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted run must not leave stale shards that contaminate the next plot.
    for stale in out.glob("shard_*.pt"): stale.unlink()
    model = build_encoder(args.encoder, args.device, args.checkpoint)
    shard, images, labels, frames, episodes = [], [], [], [], []
    shard_id = 0; total = 0; stride = max(1, args.stride)
    for task_id, task in enumerate(tasks):
        vids = video_files(root / task, camera, limit)
        if not vids:
            raise FileNotFoundError(f"No videos found for task={task}, camera={camera}, root={root / task}")
        for ep_id, vp in enumerate(tqdm(vids, desc=f"{args.dataset}:{task}")):
            cap = cv2.VideoCapture(str(vp)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); stride = max(1, args.stride)
            for fi in range(n):
                ok, bgr = cap.read()
                if not ok: break
                if fi % stride: continue
                images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))); labels.append(task_id); frames.append(fi); episodes.append(ep_id)
                if len(images) >= args.batch_size:
                    x = model.prepare_dino_input([[im] for im in images]); z = model(x).detach().cpu().to(torch.float16)
                    torch.save({"tokens": z, "task": np.asarray(labels,np.int16), "frame": np.asarray(frames,np.int32), "episode": np.asarray(episodes,np.int32)}, out/f"shard_{shard_id:06d}.pt")
                    total += z.shape[0]; shard_id += 1; images.clear(); labels.clear(); frames.clear(); episodes.clear()
            cap.release()
    if images:
        z = model(model.prepare_dino_input([[im] for im in images])).detach().cpu().to(torch.float16)
        torch.save({"tokens": z, "task": np.asarray(labels,np.int16), "frame": np.asarray(frames,np.int32), "episode": np.asarray(episodes,np.int32)}, out/f"shard_{shard_id:06d}.pt"); total += z.shape[0]
    (out/"manifest.json").write_text(json.dumps({"dataset":args.dataset,"encoder":args.encoder,"tasks":tasks,"stride":stride,"frames":total},indent=2))
    print(f"saved {total} frames to {out}")

def plot(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    _, tasks, _, _ = dataset_spec(args.dataset); cache = OUT/"cache"/args.dataset/(args.name or (args.encoder + "_ft" if args.checkpoint else args.encoder))
    rng=np.random.default_rng(args.seed); samples=[None for _ in tasks]; seen=np.zeros(len(tasks),dtype=np.int64)
    files=sorted(cache.glob("shard_*.pt"))
    if not files: raise FileNotFoundError(f"No cache shards found: {cache}")
    for f in tqdm(files, desc=f"loading {args.dataset}:{args.encoder}"):
        # Cache files are generated locally by this tool; weights_only=False is
        # needed for the NumPy metadata arrays written by older runs.
        d=torch.load(f,map_location="cpu",weights_only=False); z=d["tokens"].float().numpy(); y=np.asarray(d["task"])
        for t in range(len(tasks)):
            ix=np.flatnonzero(y==t)
            if not ix.size: continue
            cand=z[ix].reshape(-1,z.shape[-1]); seen[t] += len(cand)
            cur=samples[t]
            if cur is None: cur=cand[:args.points_per_task].copy()
            else:
                pool=np.concatenate([cur,cand],axis=0)
                take=min(args.points_per_task,len(pool)); cur=pool[rng.choice(len(pool),take,replace=False)]
            samples[t]=cur
    xs=[]; ys=[]
    for t,a in enumerate(samples):
        if a is not None and len(a): xs.append(a); ys.extend([t]*len(a))
    if not xs: raise RuntimeError(f"Cache contains no task tokens: {cache}")
    x=np.concatenate(xs); y=np.asarray(ys); x=PCA(n_components=min(args.pca_dim,x.shape[1],len(x)-1),random_state=args.seed).fit_transform(x)
    emb=TSNE(n_components=2, perplexity=min(args.perplexity,max(5,(len(x)-1)//3)), init="pca", learning_rate="auto", random_state=args.seed, max_iter=args.max_iter).fit_transform(x)
    plt.figure(figsize=(15,10)); colors=plt.get_cmap("tab20b")(np.linspace(.02,.98,len(tasks)))[:,:3]; handles=[]
    for t,task in enumerate(tasks):
        m=y==t
        if m.any():
            plt.scatter(emb[m,0],emb[m,1],s=args.point_size,alpha=args.alpha,color=colors[t],linewidths=0,rasterized=True)
            handles.append(Line2D([0],[0],marker="o",color="none",markerfacecolor=colors[t],markeredgecolor=colors[t],markersize=7,label=task))
    plt.title(f"{args.encoder} — {args.dataset} token t-SNE",pad=14,fontweight="bold",fontsize=18); plt.xlabel("t-SNE 1",fontsize=16,labelpad=10); plt.ylabel("t-SNE 2",fontsize=16,labelpad=10); plt.tick_params(axis="both",which="major",labelsize=13)
    leg=plt.legend(handles=handles,title="Task",bbox_to_anchor=(1.02,1),loc="upper left",fontsize=8,title_fontsize=10,frameon=True,framealpha=.92,facecolor="white",edgecolor="#bbbbbb",borderpad=.8,handletextpad=.5)
    leg.get_frame().set_linewidth(.8); plt.grid(False); plt.tight_layout()
    out=OUT/"figures"; out.mkdir(parents=True,exist_ok=True); plt.savefig(out/f"{args.dataset}_{(args.name or args.encoder)}.png",dpi=300); plt.savefig(out/f"{args.dataset}_{(args.name or args.encoder)}.pdf"); plt.close()


def modality_specs():
    return {
        "robotwin_cam_high": (ROBOTWIN_ROOT, ROBOTWIN_TASKS, "cam_high", 500),
        "robotwin_cam_left_wrist": (ROBOTWIN_ROOT, ROBOTWIN_TASKS, "cam_left_wrist", 500),
        "robotwin_cam_right_wrist": (ROBOTWIN_ROOT, ROBOTWIN_TASKS, "cam_right_wrist", 500),
        "robocasa_ego_view": (ROBOCASA_ROOT, ROBOCASA_TASKS, "ego_view", 1000),
    }

def _sample_modality_values(root, tasks, limit, field, stride):
    import pandas as pd
    vals=[]; labels=[]
    for tid, task in enumerate(tasks):
        files=sorted((root/task).glob("data/chunk-*/episode_*.parquet"))[:limit]
        if not files: raise FileNotFoundError(f"No parquet episodes for task={task} under {root}")
        for f in tqdm(files, desc=f"{field}:{task}", leave=False):
            d=pd.read_parquet(f, columns=[field])
            a=np.asarray(d[field].tolist(), dtype=np.float32)
            if a.ndim != 2: raise ValueError(f"{f}: {field} is not a 2D vector column: {a.shape}")
            x=a[::stride]
            vals.append(x); labels.extend([tid]*len(x))
    return np.concatenate(vals,axis=0), np.asarray(labels,np.int16)

def plot_modality(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    spec=modality_specs()[args.view]; root,tasks,_,limit=spec
    x,y=_sample_modality_values(root,tasks,limit,args.field,args.stride)
    rng=np.random.default_rng(args.seed); xs=[]; ys=[]
    for t in range(len(tasks)):
        z=x[y==t];
        if len(z)>args.points_per_task: z=z[rng.choice(len(z),args.points_per_task,replace=False)]
        if len(z): xs.append(z); ys.extend([t]*len(z))
    if not xs: raise RuntimeError("No modality samples")
    x=np.concatenate(xs); y=np.asarray(ys)
    # Standardize dimensions so joint/gripper scales do not dominate the
    # geometry; remove constant dimensions before PCA/t-SNE.
    x=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu=x.mean(axis=0, keepdims=True); sd=x.std(axis=0, keepdims=True)
    keep=(sd[0] > 1e-8)
    if keep.any(): x=(x[:,keep]-mu[:,keep])/(sd[:,keep]+1e-8)
    else: x=x-mu
    x=PCA(n_components=min(args.pca_dim,x.shape[1],len(x)-1),random_state=args.seed).fit_transform(x)
    perp=min(args.perplexity,max(5,(len(x)-1)//3))
    emb=TSNE(n_components=2,perplexity=perp,init="pca",learning_rate="auto",random_state=args.seed,max_iter=args.max_iter).fit_transform(x)
    fig,ax=plt.subplots(figsize=(15,10)); colors=plt.get_cmap("tab20b")(np.linspace(.02,.98,len(tasks)))[:,:3]; handles=[]
    for t,task in enumerate(tasks):
        m=y==t
        if m.any():
            ax.scatter(emb[m,0],emb[m,1],s=args.point_size,alpha=args.alpha,color=colors[t],linewidths=0,rasterized=True)
            handles.append(Line2D([0],[0],marker="o",color="none",markerfacecolor=colors[t],markeredgecolor=colors[t],markersize=7,label=task))
    ax.set_title(f"{args.field} — {args.view} t-SNE",fontsize=18,fontweight="bold",pad=14); ax.set_xlabel("t-SNE 1",fontsize=16,labelpad=10); ax.set_ylabel("t-SNE 2",fontsize=16,labelpad=10); ax.tick_params(axis="both",which="major",labelsize=13)
    leg=ax.legend(handles=handles,title="Task",bbox_to_anchor=(1.02,1),loc="upper left",fontsize=8,title_fontsize=10,frameon=True,framealpha=.92,facecolor="white",edgecolor="#bbbbbb"); leg.get_frame().set_linewidth(.8); fig.tight_layout()
    out=OUT/"figures"; out.mkdir(parents=True,exist_ok=True); stem=f"{args.view}_{args.field.replace('observation.','').replace('.','_')}"; fig.savefig(out/f"{stem}.png",dpi=300); fig.savefig(out/f"{stem}.pdf"); plt.close(fig)

VLM_MODELS = {
    "depthvlm": str(ROOT / "playground/Pretrained_models/DepthVLM-4B"),
    "qwen3": str(ROOT / "playground/Pretrained_models/Qwen3-VL-4B-Instruct"),
    "xiaomi": str(ROOT / "playground/Pretrained_models/Xiaomi-Robotics"),
}

def build_vlm(name, device):
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "Policy"))
    cfg=OmegaConf.create({"framework":{"qwenvl":{"base_vlm":VLM_MODELS[name]}},"datasets":{"vla_data":{}}})
    from hivebench.model.modules.vlm import get_vlm_model
    m=get_vlm_model(cfg).eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.inference_mode()
def extract_vlm(args):
    root,tasks,camera,limit=dataset_spec(args.dataset); out=OUT/"cache"/args.dataset/args.name; out.mkdir(parents=True,exist_ok=True)
    for stale in out.glob("shard_*.pt"): stale.unlink()
    model=build_vlm(args.vlm,args.device); images=[]; labels=[]; frames=[]; episodes=[]; sid=0; total=0
    for tid,task in enumerate(tasks):
      vids=video_files(root/task,camera,limit)
      for eid,vp in enumerate(tqdm(vids,desc=f"{args.dataset}:{task}")):
       cap=cv2.VideoCapture(str(vp)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       for fi in range(n):
        ok,bgr=cap.read()
        if not ok: break
        if fi%args.stride: continue
        images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))); labels.append(tid); frames.append(fi); episodes.append(eid)
        if len(images)>=args.batch_size:
         z=model.encode_visual_tokens([[im] for im in images],[""]*len(images)).detach().cpu().to(torch.float16)
         torch.save({"tokens":z,"task":np.asarray(labels,np.int16),"frame":np.asarray(frames,np.int32),"episode":np.asarray(episodes,np.int32)},out/f"shard_{sid:06d}.pt"); sid+=1; total+=z.shape[0]; images.clear(); labels.clear(); frames.clear(); episodes.clear()
       cap.release()
    if images:
      z=model.encode_visual_tokens([[im] for im in images],[""]*len(images)).detach().cpu().to(torch.float16); torch.save({"tokens":z,"task":np.asarray(labels,np.int16),"frame":np.asarray(frames,np.int32),"episode":np.asarray(episodes,np.int32)},out/f"shard_{sid:06d}.pt"); total+=z.shape[0]
    (out/"manifest.json").write_text(json.dumps({"dataset":args.dataset,"vlm":args.vlm,"frames":total},indent=2))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True); common=argparse.ArgumentParser(add_help=False); common.add_argument("--dataset",choices=["robotwin","robocasa"],required=True); common.add_argument("--encoder",choices=ENCODERS,required=True); common.add_argument("--checkpoint",default=None); common.add_argument("--name",default=None)
    e=sub.add_parser("extract",parents=[common]); e.add_argument("--device",default="cuda"); e.add_argument("--batch-size",type=int,default=16); e.add_argument("--stride",type=int,default=10); e.set_defaults(func=extract)
    v=sub.add_parser("vlm",parents=[]); v.add_argument("--vlm",choices=list(VLM_MODELS),required=True); v.add_argument("--dataset",choices=["robotwin","robocasa"],required=True); v.add_argument("--name",required=True); v.add_argument("--device",default="cuda"); v.add_argument("--batch-size",type=int,default=4); v.add_argument("--stride",type=int,default=10); v.set_defaults(func=extract_vlm)
    m=sub.add_parser("modality",parents=[]); m.add_argument("--view",choices=list(modality_specs()),required=True); m.add_argument("--field",choices=["action","observation.state"],required=True); m.add_argument("--stride",type=int,default=10); m.add_argument("--points-per-task",type=int,default=5000); m.add_argument("--pca-dim",type=int,default=20); m.add_argument("--perplexity",type=float,default=30); m.add_argument("--max-iter",type=int,default=1000); m.add_argument("--point-size",type=float,default=2.5); m.add_argument("--alpha",type=float,default=.65); m.add_argument("--seed",type=int,default=42); m.set_defaults(func=plot_modality)
    q=sub.add_parser("plot",parents=[common]); q.add_argument("--points-per-task",type=int,default=5000); q.add_argument("--pca-dim",type=int,default=50); q.add_argument("--perplexity",type=float,default=30); q.add_argument("--max-iter",type=int,default=1000); q.add_argument("--point-size",type=float,default=3); q.add_argument("--alpha",type=float,default=.8); q.add_argument("--seed",type=int,default=42); q.set_defaults(func=plot)
    a=sub.add_parser("all"); a.add_argument("--dataset",choices=["robotwin","robocasa","both"],default="both"); a.add_argument("--device",default="cuda"); a.add_argument("--batch-size",type=int,default=16); a.add_argument("--points-per-task",type=int,default=5000); a.set_defaults(func=None)
    args=p.parse_args()
    if args.cmd=="all":
        for d in (["robotwin","robocasa"] if args.dataset=="both" else [args.dataset]):
            for enc in ENCODERS:
                extract(argparse.Namespace(dataset=d,encoder=enc,device=args.device,batch_size=args.batch_size,stride=10)); plot(argparse.Namespace(dataset=d,encoder=enc,points_per_task=args.points_per_task,pca_dim=50,perplexity=30,max_iter=1000,point_size=3,alpha=.8,seed=42))
    else: args.func(args)
if __name__ == "__main__": main()
