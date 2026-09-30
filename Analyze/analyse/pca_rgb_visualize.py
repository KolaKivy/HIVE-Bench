"""Single-frame patch-token PCA-RGB visualization.
Example:
  python -m Analyze.analyse.pca_rgb_visualize --dataset robotwin --encoder clip \
    --task adjust_bottle --episode 3 --frame 120
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT=ROOT/'playground/RoboTwin_LeRobot/Randomized'
ROBOCASA_ROOT=ROOT/'playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim'
ROBOTWIN_TASKS=["adjust_bottle","beat_block_hammer","click_alarmclock","handover_block","lift_pot","move_playingcard_away","open_laptop","place_burger_fries","place_can_basket","rotate_qrcode","stamp_seal","turn_switch"]
ROBOCASA_TASKS=["gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000","gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"]
CAMERAS={'cam_high':'observation.images.cam_high','cam_left_wrist':'observation.images.cam_left_wrist','cam_right_wrist':'observation.images.cam_right_wrist','ego_view':'observation.images.ego_view'}

def video_path(dataset,task,episode,camera):
 root=ROBOTWIN_ROOT if dataset=='robotwin' else ROBOCASA_ROOT
 c=CAMERAS[camera]; base=root/task/'videos'
 files=sorted(base.glob(f'chunk-*/{c}/*.mp4'))
 if not files: files=sorted(base.glob(f'chunk-*/{camera}/*.mp4'))
 if episode<0 or episode>=len(files): raise IndexError(f'episode {episode} out of range (found {len(files)} videos)')
 return files[episode]

def load_model(name,device,checkpoint=None):
 sys.path.insert(0,str(ROOT/'Policy'))
 from hivebench.model.framework.DinoGR00T import _build_vision_encoder
 m=_build_vision_encoder(name).eval().to(device)
 if checkpoint:
  from Analyze.analyse.idm_fdm_model import load_finetuned_vision_encoder
  load_finetuned_vision_encoder(m,checkpoint)
 return m

def patch_tokens(model,image,device):
 with torch.inference_mode():
  x=model.prepare_dino_input([[image]])
  z=model(x)
 z=z.detach().float().cpu()
 if z.ndim==4: z=z[0].flatten(0,1)
 elif z.ndim==3: z=z[0]
 elif z.ndim==2: pass
 else: raise ValueError(f'Unexpected encoder output shape {tuple(z.shape)}')
 n,d=z.shape; side=int(round((n-1)**0.5))
 if side*side==n-1: z=z[1:]; side=int(round(len(z)**0.5))
 elif int(round(n**0.5))**2==n: side=int(round(n**0.5))
 else: raise ValueError(f'Token count {n} is not square after CLS removal')
 return z.numpy().reshape(side,side,d)

def pca_rgb(tokens):
 from sklearn.decomposition import PCA
 h,w,d=tokens.shape; flat=tokens.reshape(-1,d); q=PCA(n_components=min(3,d,len(flat)),random_state=0).fit_transform(flat)
 out=np.zeros((len(flat),3),np.float32)
 for k in range(3):
  v=q[:,k] if k<q.shape[1] else np.zeros(len(flat))
  lo,hi=np.percentile(v,[1,99]); out[:,k]=np.clip((v-lo)/(hi-lo+1e-8),0,1)
 out=np.power(out,0.8); return out.reshape(h,w,3)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['robotwin','robocasa'],required=True); ap.add_argument('--encoder',required=True); ap.add_argument('--checkpoint'); ap.add_argument('--task',required=True); ap.add_argument('--episode',type=int,required=True); ap.add_argument('--frame',type=int,required=True); ap.add_argument('--camera',choices=list(CAMERAS)); ap.add_argument('--device',default='cuda'); ap.add_argument('--output-dir',default=str(ROOT/'Analyze/outputs/tsne/pca_rgb')); a=ap.parse_args()
 cams=[a.camera] if a.camera else (['cam_high','cam_left_wrist','cam_right_wrist'] if a.dataset=='robotwin' else ['ego_view'])
 model=load_model(a.encoder,a.device,a.checkpoint)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 for cam in cams:
  vp=video_path(a.dataset,a.task,a.episode,cam); cap=cv2.VideoCapture(str(vp)); cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame); ok,bgr=cap.read(); cap.release()
  if not ok: raise RuntimeError(f'cannot read frame {a.frame} from {vp}')
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); im=Image.fromarray(rgb); tok=patch_tokens(model,im,a.device); vis=pca_rgb(tok)
  vis=cv2.resize((vis*255).astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
  # Preserve scene content while showing PCA representation as a heatmap-like overlay.
  overlay=cv2.addWeighted(rgb.astype(np.uint8),0.58,vis.astype(np.uint8),0.42,0)
  stem=f'{a.dataset}_{a.encoder}_{a.task}_ep{a.episode}_frame{a.frame}_{cam}'; token_path=out/(stem+'_token.png'); overlay_path=out/(stem+'_overlay.png'); Image.fromarray(vis).save(token_path); Image.fromarray(overlay).save(overlay_path); np.save(out/(stem+'_tokens.npy'),tok)
  print(f'saved token={token_path} overlay={overlay_path} shape={tok.shape} source={vp}',flush=True)
if __name__=='__main__': main()
