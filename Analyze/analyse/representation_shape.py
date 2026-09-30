from __future__ import annotations
import argparse, json, math, os, random, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'Analyze/outputs/representation_shape'
ROBOTWIN=ROOT/'playground/RoboTwin_LeRobot_HeadCam/Randomized'; ROBOCASA=ROOT/'playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim'
RTASKS=['adjust_bottle','beat_block_hammer','click_alarmclock','handover_block','lift_pot','move_playingcard_away','open_laptop','place_burger_fries','place_can_basket','rotate_qrcode','stamp_seal','turn_switch']
CTASKS=['gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000','gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000']
BASE=['dinov2_base','dinov3_base','spa_base','vc1_base','clip','siglip','siglip2','mae','vit','radio','cradio','theia','internvit','voltron','vggt_omega','vjepa2_base','lingbot_large']
VLM=['depthvlm','qwen3','xiaomi']
FT_CHECKPOINTS={
 'dinov2_base_ft':str(ROOT/'playground/Checkpoints/all_task_dinov2_base_ft_robotwin_new/final_model/pytorch_model.pt'),
 'dinov3_base_ft':str(ROOT/'playground/Checkpoints/all_task_dinov3_base_ft_robotwin_new/final_model/pytorch_model.pt'),
 'spa_base_ft':str(ROOT/'playground/Checkpoints/all_task_spa_base_ft_robotwin_new/final_model/pytorch_model.pt'),
 'vc1_base_ft':str(ROOT/'playground/Checkpoints/all_task_vc1_base_ft_robotwin_new/final_model/pytorch_model.pt'),
 'vggt_omega_ft':str(ROOT/'playground/Checkpoints/all_task_vggt_omega_ft_robotwin_new/final_model/pytorch_model.pt'),
 'cradio_ft':str(ROOT/'playground/Checkpoints/all_task_cradio-ft_robotwin_new/final_model/pytorch_model.pt'),
 'siglip2_ft':str(ROOT/'playground/Checkpoints/all_task_siglip2_ft_robotwin_new/final_model/pytorch_model.pt'),
 }

FT_BASE={k:'_'.join(k.split('_')[:-1]) for k in FT_CHECKPOINTS}

def samples(dataset,per_task=2500,stride=10,seed=2026):
 root,tasks,cam,limit=(ROBOTWIN,RTASKS,'observation.images.cam_high',500) if dataset=='robotwin' else (ROBOCASA,CTASKS,'observation.images.ego_view',1000)
 rng=random.Random(seed); out=[]
 for tid,task in enumerate(tasks):
  vids=sorted((root/task/'videos').glob(f'chunk-*/{cam}/*.mp4'))[:limit]
  cand=[]
  for ei,v in enumerate(vids):
   c=cv2.VideoCapture(str(v)); n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release(); cand += [(ei,fi) for fi in range(0,n,stride)]
  if per_task and len(cand)>per_task:
   # Uniformly subsample the temporal candidates for bounded memory/runtime.
   idx=np.linspace(0,len(cand)-1,per_task,dtype=int); cand=[cand[i] for i in idx]
  out += [(tid,ei,fi) for ei,fi in cand]
 rng.shuffle(out); return out,root,tasks,cam

def read_image(path,frame):
 c=cv2.VideoCapture(str(path)); c.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,b=c.read(); c.release()
 if not ok: return None
 return Image.fromarray(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))

def norm_grid(z,target=14):
 if z.ndim==3: z=z[0]
 n,d=z.shape
 s=math.isqrt(n)
 if s*s!=n:
  if math.isqrt(n-1)**2==n-1: z=z[1:]; s=math.isqrt(len(z))
  else:
   # Explicit fallback for non-spatial VLM token sequences: deterministic 1-D pooling.
   x=torch.from_numpy(z.T).unsqueeze(0); z=torch.nn.functional.adaptive_avg_pool1d(x,target*target).squeeze(0).T.numpy(); s=target
 if s!=target:
  x=torch.from_numpy(z.reshape(s,s,d).transpose(2,0,1)).unsqueeze(0); z=torch.nn.functional.adaptive_avg_pool2d(x,(target,target)).squeeze(0).permute(1,2,0).reshape(-1,d).numpy()
 return z

def load_enc(name,device):
 sys.path.insert(0,str(ROOT/'Policy')); from hivebench.model.framework.DinoGR00T import _build_vision_encoder
 base=FT_BASE.get(name,name); model=_build_vision_encoder(base).eval().to(device)
 if name in FT_CHECKPOINTS:
  from Analyze.analyse.idm_fdm_model import load_finetuned_vision_encoder
  load_finetuned_vision_encoder(model,FT_CHECKPOINTS[name])
 return model
def load_vlm(name,device):
 sys.path.insert(0,str(ROOT/'Policy')); from Analyze.analyse.tsne_analysis import build_vlm
 return build_vlm(name,device)

def extract(args,items,root,cam):
 cache=OUT/'cache'/args.dataset/args.name; cache.mkdir(parents=True,exist_ok=True); path=cache/'features.pt'
 if path.exists() and not args.force:
  cached=torch.load(path,map_location='cpu',weights_only=False)
  # Reuse only when the sampling protocol is identical; otherwise stale caches
  # (e.g. the former 2500-per-task run) would silently invalidate the experiment.
  if int(cached.get('num_samples',-1)) == len(items) and int(cached.get('stride',-1)) == int(args.stride):
   return cached
  print(f'cache sampling mismatch: cached={cached.get("num_samples")} requested={len(items)}; re-extracting')
 model=load_vlm(args.name,args.device) if args.name in VLM else load_enc(args.name,args.device)
 projected=[]; jl=None; source_dim=None
 for st in tqdm(range(0,len(items),args.encode_batch_size),desc=f'extract {args.dataset}:{args.name}'):
  batch=items[st:st+args.encode_batch_size]; imgs=[]
  for tid,ei,fi in batch:
   task=(RTASKS if args.dataset=='robotwin' else CTASKS)[tid]; views=[cam]
   if args.dataset=='robotwin': views=['observation.images.cam_high']
   ims=[read_image(sorted((root/task/'videos').glob(f'chunk-*/{v}/*.mp4'))[ei],fi) for v in views]
   if any(im is None for im in ims):
    continue
   imgs.append(ims)
  if args.name in VLM:
   rows=[]
   with torch.inference_mode():
    for ims in imgs:
     vv=[norm_grid(model.encode_visual_tokens([[im]],['']).detach().cpu().float().numpy()) for im in ims]; rows.append(np.concatenate(vv,0))
  else:
   with torch.inference_mode():
    x=model.prepare_dino_input(imgs); z=model(x).detach().cpu().float().numpy()
   nv=len(imgs[0]); rows=[np.concatenate([norm_grid(z[i*nv+j]) for j in range(nv)],0) for i in range(len(imgs))]
  if not rows: continue
  raw=np.stack(rows).astype('float32')
  if jl is None:
   source_dim=raw.shape[-1]; g=torch.Generator().manual_seed(2026); jl=(torch.randn(source_dim,256,generator=g)/math.sqrt(256)).numpy()
  projected.append(np.einsum('bnd,dk->bnk',raw,jl).astype('float16'))
  del raw, rows, imgs
 tok=np.concatenate(projected,axis=0); del projected
 payload={'encoder':args.name,'dataset':args.dataset,'tokens':torch.from_numpy(tok),'num_samples':len(items),'num_views':tok.shape[1]//196,'tokens_per_view':196,'source_dim':source_dim,'common_dim':256,'sample_seed':2026,'stride':args.stride,'items':items,'jl_seed':2026}
 torch.save(payload,path); del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None; return payload

def train(args,payload):
 from Analyze.analyse.representation_shape_core import WeakAdapter,Shape,target_discrepancy,full_loss
 x=payload['tokens'].float(); n=x.shape[0]; model=WeakAdapter(input_dim=256).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); src=x.mean(1); init=None; rows=[]; best=None
 for ep in range(args.epochs):
  for st in range(0,n,args.batch_size):
   t=x[st:st+args.batch_size].to(args.device); s=src[st:st+args.batch_size].to(args.device); z=model(t); shape=Shape(args.target); disc=target_discrepancy(z,shape); init=init or float(disc.detach()); loss=full_loss(z,s,shape,init); opt.zero_grad(); loss.backward(); opt.step(); rows.append((len(rows),float(loss),float(disc)))
  if best is None or rows[-1][1]<best: best=rows[-1][1]
 out=OUT/'results'/args.dataset/args.target/args.name; out.mkdir(parents=True,exist_ok=True); import csv
 with open(out/'metrics.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['step','loss','target_loss']); w.writerows(rows)
 # Render the training curves for visual inspection.
 try:
  import matplotlib; matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  steps=np.asarray([r[0] for r in rows]); losses=np.asarray([r[1] for r in rows]); targets=np.asarray([r[2] for r in rows])
  fig,ax=plt.subplots(figsize=(8,4.5),dpi=180)
  ax.plot(steps,losses,label='total loss',lw=1.5)
  ax.plot(steps,targets,label='target discrepancy',lw=1.2,alpha=.85)
  ax.set_xlabel('Training step'); ax.set_ylabel('Loss'); ax.set_title(f'{args.dataset} / {args.name} / {args.target}')
  ax.grid(True,alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'loss_curve.png'); plt.close(fig)
 except Exception as e:
  print(f'warning: could not render loss_curve.png: {e}', file=sys.stderr)
 vals=np.asarray([r[1] for r in rows])
 with torch.no_grad():
  pdelta=float(torch.sqrt(sum((p.detach()**2).sum() for p in model.parameters())).cpu())
  # Evaluate final structure in chunks; never move the full dataset to GPU.
  zparts=[]
  for st in range(0, n, args.batch_size):
   zparts.append(model(x[st:st+args.batch_size].to(args.device)).detach().cpu())
  zfinal=torch.cat(zparts,0); src_cpu=src
  from scipy.stats import spearmanr
  # Pairwise distances are quadratic; use a deterministic subset for large runs.
  metric_n=min(len(zfinal),2048); zmetric=zfinal[:metric_n]; smetric=src_cpu[:metric_n]
  distance_spearman=float(spearmanr(torch.pdist(zmetric).numpy(),torch.pdist(smetric).numpy()).statistic)
  rel=vals/abs(vals[0]); i25=int(np.argmax(rel<=0.25)) if np.any(rel<=0.25) else -1; i10=int(np.argmax(rel<=0.10)) if np.any(rel<=0.10) else -1
 summary={'dataset':args.dataset,'encoder':args.name,'target_shape':args.target,'seed':args.seed,'num_samples':payload['num_samples'],'num_views':payload['num_views'],'tokens_per_view':196,'common_dim':256,'normalized_auc':float(np.trapz(rel,dx=1)/len(vals)),'steps_to_25pct':i25,'steps_to_10pct':i10,'target_discrepancy':float(rows[-1][2]),'parameter_delta_ratio':pdelta,'distance_spearman':distance_spearman,'steps':len(rows)}; (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(out)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['robotwin','robocasa'],required=True); ap.add_argument('--name',required=True); ap.add_argument('--target',choices=['gaussian','sphere'],required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--encode-batch-size',type=int,default=8); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--points-per-task',type=int,default=-1, help='max sampled frames per task; -1=dataset default (robotwin 2000, robocasa 500), 0=all'); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--force',action='store_true'); a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); a.name=a.name; a.points_per_task = (2000 if a.dataset=='robotwin' else 500) if a.points_per_task < 0 else a.points_per_task; items,root,tasks,cam=samples(a.dataset,a.points_per_task,a.stride,a.seed); payload=extract(a,items,root,cam); train(a,payload)
if __name__=='__main__': main()
