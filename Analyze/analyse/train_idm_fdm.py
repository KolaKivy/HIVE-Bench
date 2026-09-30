"""Unified IDM/FDM training entrypoint for the frozen HIVE-Bench vision encoders."""
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .idm_fdm_dataset import (CONTROL_ACTION_INDICES, CONTROL_GROUPS, ROBOTWIN_ACTION_INDICES, ROBOTWIN_CONTROL_GROUPS, FDMDataset, IDMDataset, StateDataset, RoboTwinObjectDataset, collate_fn, fdm_collate_fn, state_collate_fn, object_collate_fn)
from .idm_fdm_model import FDMModel, IDMModel, StateModel, ObjectStateModel, build_encoder, load_finetuned_vision_encoder


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _idm_metrics(pred, target, groups=CONTROL_GROUPS):
    prs=[]
    for i in range(target.shape[1]):
        if target[:,i].std() < 1e-8 or pred[:,i].std() < 1e-8: prs.append(0.0)
        else:
            r,_=pearsonr(pred[:,i],target[:,i]); prs.append(0.0 if np.isnan(r) else float(r))
    out={"MSE":float(((pred-target)**2).mean()),"L1":float(np.abs(pred-target).mean()),"pearson_r_mean":float(np.mean(prs)),"per_dim_L1":np.abs(pred-target).mean(0).tolist(),"per_dim_pearson":prs}
    for k,ix in groups.items(): out[k+"_L1"]=float(np.abs(pred[:,ix]-target[:,ix]).mean()); out[k+"_pearson"]=float(np.mean(np.asarray(prs)[ix]))
    return out


def eval_idm(model, loader, mean, std, groups=CONTROL_GROUPS, limit=None):
    model.eval(); p=[]; y=[]; tasks=[]
    with torch.no_grad():
        for step,(x,xh,target,bt) in enumerate(tqdm(loader,desc="  IDM eval",leave=False,dynamic_ncols=True)):
            if limit is not None and step >= limit: break
            p.append(model(x,xh)["pred_action"].cpu().float().numpy()); y.append(target.float().numpy()); tasks+=bt
    p,y=np.concatenate(p),np.concatenate(y); nm=_idm_metrics(p,y,groups); rawp,rawy=p*std+mean,y*std+mean; rm=_idm_metrics(rawp,rawy,groups)
    out={"normalized_MSE":nm["MSE"],"normalized_L1":nm["L1"],"raw_MSE":rm["MSE"],"raw_L1":rm["L1"],"pearson_r_mean":rm["pearson_r_mean"],"per_dim_raw_L1":rm["per_dim_L1"],"per_dim_pearson":rm["per_dim_pearson"]}
    for k in groups: out[k+"_L1"]=rm[k+"_L1"]; out[k+"_pearson"]=rm[k+"_pearson"]
    per={}
    for t in sorted(set(tasks)):
        ix=np.asarray([z==t for z in tasks]); per[t]=_idm_metrics(rawp[ix],rawy[ix],groups)
    out["per_task"]=per; out["macro_task_raw_L1"]=float(np.mean([x["L1"] for x in per.values()])); out["macro_task_pearson_r"]=float(np.mean([x["pearson_r_mean"] for x in per.values()]))
    return out


def eval_fdm(model, loader, limit=None):
    model.eval(); n=0; mse=0.; cos=0.; task=defaultdict(lambda:[0,0.,0.])
    with torch.no_grad():
        for step,(x,xh,a,bt) in enumerate(tqdm(loader,desc="  FDM eval",leave=False,dynamic_ncols=True)):
            if limit is not None and step >= limit: break
            o=model(x,xh,a); b=len(bt); m=o["loss_mse"].item(); c=o["cosine_similarity"].item(); n+=b; mse+=m*b; cos+=c*b
            for t in set(bt):
                k=sum(z==t for z in bt); task[t][0]+=k; task[t][1]+=m*k; task[t][2]+=c*k
    per={t:{"normalized_feature_MSE":v[1]/v[0],"token_cosine_similarity":v[2]/v[0]} for t,v in task.items()}
    return {"normalized_feature_MSE":mse/n,"token_cosine_similarity":cos/n,"per_task":per,"macro_task_normalized_feature_MSE":float(np.mean([x["normalized_feature_MSE"] for x in per.values()])),"macro_task_token_cosine_similarity":float(np.mean([x["token_cosine_similarity"] for x in per.values()]))}


def dataset_type(cfg):
    return str(cfg.get("dataset_type", "robocasa")).lower()


def groups_for(cfg):
    return ROBOTWIN_CONTROL_GROUPS if dataset_type(cfg) == "robotwin" else CONTROL_GROUPS


def common(cfg):
    is_robotwin = dataset_type(cfg) == "robotwin"
    indices = ROBOTWIN_ACTION_INDICES if is_robotwin else CONTROL_ACTION_INDICES
    return dict(
        data_root=cfg.data_root,
        dataset_names=list(cfg.dataset_names),
        camera_key=cfg.camera_key,
        horizon=cfg.horizon,
        train_episodes=cfg.train_episodes,
        val_episodes=cfg.val_episodes,
        seed=cfg.seed,
        action_indices=indices,
    )


def finetuned_encoder_checkpoint(cfg):
    value = str(cfg.get("finetuned_encoder_checkpoint") or "").strip()
    return hydra.utils.to_absolute_path(value) if value else None


def encoder_label(cfg):
    base = str(cfg.encoder_name).replace("/", "_")
    checkpoint = finetuned_encoder_checkpoint(cfg)
    if checkpoint is None:
        return base
    path = Path(checkpoint)
    run_name = path.parent.parent.name if path.parent.name == "final_model" else path.stem
    return f"{base}_ft_{run_name}".replace("/", "_")


def make_model(cfg, device, fdm=False):
    # VLM aliases such as qwen3_layer16 encode the selected hidden layer;
    # configs may also provide an explicit vlm_layer_idx override.
    enc=build_encoder(cfg.encoder_name, int(cfg.get("vlm_layer_idx", -1)))
    checkpoint=finetuned_encoder_checkpoint(cfg)
    if checkpoint is not None:
        load_finetuned_vision_encoder(enc, checkpoint)
    enc=enc.to(device)
    if fdm == "state": return StateModel(enc,state_dim=cfg.action_dim,enc_dim=cfg.get("encoder_dim"),num_layers=cfg.num_layers,num_heads=cfg.num_heads,dropout=cfg.dropout).to(device)
    if fdm == "object": return ObjectStateModel(enc,num_queries=cfg.object_num_queries,enc_dim=cfg.get("encoder_dim"),num_layers=cfg.num_layers,num_heads=cfg.num_heads,dropout=cfg.dropout).to(device)
    if fdm: return FDMModel(enc,action_dim=cfg.action_dim,enc_dim=cfg.get("encoder_dim"),model_dim=cfg.model_dim,num_layers=cfg.num_layers,num_heads=cfg.num_heads,dropout=cfg.dropout).to(device)
    return IDMModel(enc,action_dim=cfg.action_dim,enc_dim=cfg.get("encoder_dim"),head_type=cfg.head_type,model_dim=cfg.model_dim,num_layers=cfg.num_layers,num_heads=cfg.num_heads,hidden_dim=cfg.hidden_dim,pool=cfg.pool,dropout=cfg.dropout).to(device)


def outdir(cfg):
    d=Path(cfg.output_dir)/(encoder_label(cfg)+f"_{cfg.mode}_h{cfg.horizon}_{cfg.head_type}"); d.mkdir(parents=True,exist_ok=True); return d


def run_idm(cfg, device):
    groups=groups_for(cfg); tr=IDMDataset(split="train",**common(cfg)); stat=tr.stats(); tr.set_normalization(stat); va=IDMDataset(split="val",normalization=stat,**common(cfg))
    mean,std=np.asarray(stat["mean"],np.float32),np.asarray(stat["std"],np.float32); model=make_model(cfg,device)
    tl=DataLoader(tr,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=collate_fn,pin_memory=True); vl=DataLoader(va,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate_fn,pin_memory=True)
    d=outdir(cfg); json.dump({"mode":"idm","dataset_type":dataset_type(cfg),"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"target":"action[t+horizon-1]-state[t]","normalization":stat,"train":tr.manifest(),"val":va.manifest()},open(d/"data_manifest.json","w"),indent=2)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr*.01); best=float("inf"); ckpt=d/"best.pt"
    with open(d/"training_log.csv","w",newline="") as f: csv.writer(f).writerow(["epoch","train_normalized_mse","val_normalized_l1","val_raw_l1","val_pearson","lr"])
    for e in range(1,cfg.epochs+1):
        model.train(); model.encoder.eval(); ls=[]; t0=time.time()
        for step,(x,xh,y,_) in enumerate(tqdm(tl,desc=f"IDM {e}/{cfg.epochs}",dynamic_ncols=True)):
            if cfg.max_steps_per_epoch is not None and step>=cfg.max_steps_per_epoch: break
            opt.zero_grad(); o=model(x,xh,y.to(device)); o["loss_mse"].backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.); opt.step(); ls.append(o["loss_mse"].item())
        sch.step(); m=eval_idm(model,vl,mean,std,groups,cfg.max_steps_per_epoch); row=[e,float(np.mean(ls)),m["normalized_L1"],m["raw_L1"],m["pearson_r_mean"],opt.param_groups[0]["lr"]]
        with open(d/"training_log.csv","a",newline="") as f: csv.writer(f).writerow(row)
        print(f"IDM epoch {e}: {time.time()-t0:.1f}s train_nmse={row[1]:.4f} val_nl1={row[2]:.4f} raw_l1={row[3]:.4f}")
        if m["normalized_L1"]<best: best=m["normalized_L1"]; torch.save(model.state_dict(),ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=device)); final=eval_idm(model,vl,mean,std,groups); final.update({"mode":"idm","encoder":cfg.encoder_name,"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"horizon":cfg.horizon,"best_normalized_val_l1":best}); json.dump(final,open(d/"results.json","w"),indent=2)


def run_fdm(cfg, device):
    tr=FDMDataset(split="train",**common(cfg)); stat=tr.action_stats(); tr.set_action_normalization(stat); va=FDMDataset(split="val",action_normalization=stat,**common(cfg)); model=make_model(cfg,device,fdm=True)
    tl=DataLoader(tr,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=fdm_collate_fn,pin_memory=True); vl=DataLoader(va,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=fdm_collate_fn,pin_memory=True)
    d=outdir(cfg); json.dump({"mode":"fdm","dataset_type":dataset_type(cfg),"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"target":"L2-normalized encoder(frame[t+horizon]) patch tokens","action_tokens":"[a_t-state_t, a_t+1-a_t, ..., a_t+15-a_t+14]","action_normalization":stat,"train":tr.manifest(),"val":va.manifest()},open(d/"data_manifest.json","w"),indent=2)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr*.01); best=float("inf"); ckpt=d/"best.pt"
    with open(d/"training_log.csv","w",newline="") as f: csv.writer(f).writerow(["epoch","train_feature_mse","val_feature_mse","val_cosine","lr"])
    for e in range(1,cfg.epochs+1):
        model.train(); model.encoder.eval(); ls=[]; t0=time.time()
        for step,(x,xh,a,_) in enumerate(tqdm(tl,desc=f"FDM {e}/{cfg.epochs}",dynamic_ncols=True)):
            if cfg.max_steps_per_epoch is not None and step>=cfg.max_steps_per_epoch: break
            opt.zero_grad(); o=model(x,xh,a); o["loss_mse"].backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.); opt.step(); ls.append(o["loss_mse"].item())
        sch.step(); m=eval_fdm(model,vl,cfg.max_steps_per_epoch); row=[e,float(np.mean(ls)),m["normalized_feature_MSE"],m["token_cosine_similarity"],opt.param_groups[0]["lr"]]
        with open(d/"training_log.csv","a",newline="") as f: csv.writer(f).writerow(row)
        print(f"FDM epoch {e}: {time.time()-t0:.1f}s train_mse={row[1]:.5f} val_mse={row[2]:.5f} cosine={row[3]:.5f}")
        if m["normalized_feature_MSE"]<best: best=m["normalized_feature_MSE"]; torch.save(model.state_dict(),ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=device)); final=eval_fdm(model,vl); final.update({"mode":"fdm","encoder":cfg.encoder_name,"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"horizon":cfg.horizon,"best_normalized_feature_mse":best}); json.dump(final,open(d/"results.json","w"),indent=2)


def eval_state(model, loader, mean, std, groups=CONTROL_GROUPS, limit=None):
    model.eval(); p=[]; y=[]; tasks=[]
    with torch.no_grad():
        for step,(x,target,bt) in enumerate(tqdm(loader,desc="  State eval",leave=False,dynamic_ncols=True)):
            if limit is not None and step >= limit: break
            p.append(model(x)["pred_state"].cpu().float().numpy()); y.append(target.float().numpy()); tasks += bt
    p,y=np.concatenate(p),np.concatenate(y); rawp,rawy=p*std+mean,y*std+mean
    per_dim=np.abs(rawp-rawy).mean(0); out={"normalized_MSE":float(((p-y)**2).mean()),"normalized_L1":float(np.abs(p-y).mean()),"raw_MAE":float(np.abs(rawp-rawy).mean()),"per_dim_raw_MAE":per_dim.tolist()}
    for k,ix in groups.items(): out[k+"_MAE"]=float(np.abs(rawp[:,ix]-rawy[:,ix]).mean())
    per={}
    for t in sorted(set(tasks)):
        ix=np.asarray([z==t for z in tasks]); per[t]={"raw_MAE":float(np.abs(rawp[ix]-rawy[ix]).mean())}
    out["per_task"]=per; out["macro_task_raw_MAE"]=float(np.mean([x["raw_MAE"] for x in per.values()]))
    return out


def run_state(cfg, device):
    base=common(cfg)
    tr=StateDataset(split="train",**base)
    stat=tr.state_stats(); tr.set_state_normalization(stat)
    va=StateDataset(split="val",state_normalization=stat,**base)
    mean,std=np.asarray(stat["mean"],np.float32),np.asarray(stat["std"],np.float32); model=make_model(cfg,device,fdm="state")
    tl=DataLoader(tr,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=state_collate_fn,pin_memory=True); vl=DataLoader(va,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=state_collate_fn,pin_memory=True)
    d=outdir(cfg); json.dump({"mode":"state","dataset_type":dataset_type(cfg),"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"target":"observation.state[t] active control dimensions","normalization":stat,"train":tr.manifest(),"val":va.manifest()},open(d/"data_manifest.json","w"),indent=2)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr*.01); best=float("inf"); ckpt=d/"best.pt"
    with open(d/"training_log.csv","w",newline="") as f: csv.writer(f).writerow(["epoch","train_normalized_mse","val_normalized_mse","val_raw_mae","lr"])
    for e in range(1,cfg.epochs+1):
        model.train(); model.encoder.eval(); ls=[]; t0=time.time()
        for step,(x,y,_) in enumerate(tqdm(tl,desc=f"State {e}/{cfg.epochs}",dynamic_ncols=True)):
            if cfg.max_steps_per_epoch is not None and step>=cfg.max_steps_per_epoch: break
            opt.zero_grad(); o=model(x,y.to(device)); o["loss_mse"].backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.); opt.step(); ls.append(o["loss_mse"].item())
        sch.step(); m=eval_state(model,vl,mean,std,groups_for(cfg),cfg.max_steps_per_epoch); row=[e,float(np.mean(ls)),m["normalized_MSE"],m["raw_MAE"],opt.param_groups[0]["lr"]]
        with open(d/"training_log.csv","a",newline="") as f: csv.writer(f).writerow(row)
        print(f"State epoch {e}: {time.time()-t0:.1f}s train_nmse={row[1]:.5f} val_nmse={row[2]:.5f} raw_mae={row[3]:.5f}")
        if m["normalized_MSE"]<best: best=m["normalized_MSE"]; torch.save(model.state_dict(),ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=device)); final=eval_state(model,vl,mean,std,groups_for(cfg)); final.update({"mode":"state","encoder":cfg.encoder_name,"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"state_dim":cfg.action_dim,"best_normalized_val_mse":best}); json.dump(final,open(d/"results.json","w"),indent=2)


def eval_object(model, loader, mean, std, limit=None):
    model.eval(); total=0; l1=0.; euclid=0.; task=defaultdict(lambda:[0,0.,0.])
    with torch.no_grad():
        for step,(x,y,mask,bt) in enumerate(tqdm(loader,desc="  Object eval",leave=False,dynamic_ncols=True)):
            if limit is not None and step >= limit: break
            pred=model(x)["pred_position"].cpu().float().numpy(); target=y.float().numpy(); m=mask.numpy().astype(bool)
            rawp,rawy=pred*std+mean,target*std+mean
            for bi,t in enumerate(bt):
                valid=m[bi]; k=int(valid.sum()); err=np.abs(rawp[bi,valid]-rawy[bi,valid]); dist=np.linalg.norm(rawp[bi,valid]-rawy[bi,valid],axis=-1)
                total+=k; l1+=err.sum(); euclid+=dist.sum(); task[t][0]+=k; task[t][1]+=err.sum(); task[t][2]+=dist.sum()
    per={t:{"position_MAE_m":v[1]/v[0]/3,"position_Euclidean_error_m":v[2]/v[0]} for t,v in task.items()}
    return {"position_MAE_m":l1/total/3,"position_Euclidean_error_m":euclid/total,"per_task":per,"macro_task_position_MAE_m":float(np.mean([x["position_MAE_m"] for x in per.values()])),"macro_task_position_Euclidean_error_m":float(np.mean([x["position_Euclidean_error_m"] for x in per.values()]))}


def run_object(cfg, device):
    if dataset_type(cfg) != "robotwin":
        raise ValueError("mode=object is only supported with dataset_type=robotwin")
    object_root=hydra.utils.to_absolute_path(str(cfg.object_data_root))
    tr=RoboTwinObjectDataset(object_root,split="train",train_per_task=cfg.object_train_per_task,val_per_task=cfg.object_val_per_task,seed=cfg.seed)
    stat=tr.stats(); tr.set_normalization(stat)
    va=RoboTwinObjectDataset(object_root,split="val",train_per_task=cfg.object_train_per_task,val_per_task=cfg.object_val_per_task,seed=cfg.seed,normalization=stat)
    mean,std=np.asarray(stat["mean"],np.float32),np.asarray(stat["std"],np.float32); model=make_model(cfg,device,fdm="object")
    tl=DataLoader(tr,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=object_collate_fn,pin_memory=True); vl=DataLoader(va,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=object_collate_fn,pin_memory=True)
    d=outdir(cfg); json.dump({"mode":"object","dataset_type":dataset_type(cfg),"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"input":"head_camera/rgb[0]","target":"initial task-object position_camera [x,y,z]","normalization":stat,"train":tr.manifest(),"val":va.manifest()},open(d/"data_manifest.json","w"),indent=2)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs,eta_min=cfg.lr*.01); best=float("inf"); ckpt=d/"best.pt"
    with open(d/"training_log.csv","w",newline="") as f: csv.writer(f).writerow(["epoch","train_normalized_mse","val_position_mae_m","val_position_euclidean_m","lr"])
    for e in range(1,cfg.epochs+1):
        model.train(); model.encoder.eval(); ls=[]; t0=time.time()
        for step,(x,y,m,_) in enumerate(tqdm(tl,desc=f"Object {e}/{cfg.epochs}",dynamic_ncols=True)):
            if cfg.max_steps_per_epoch is not None and step>=cfg.max_steps_per_epoch: break
            opt.zero_grad(); o=model(x,y.to(device),m.to(device)); o["loss_mse"].backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.); opt.step(); ls.append(o["loss_mse"].item())
        sch.step(); met=eval_object(model,vl,mean,std,cfg.max_steps_per_epoch); row=[e,float(np.mean(ls)),met["position_MAE_m"],met["position_Euclidean_error_m"],opt.param_groups[0]["lr"]]
        with open(d/"training_log.csv","a",newline="") as f: csv.writer(f).writerow(row)
        print(f"Object epoch {e}: {time.time()-t0:.1f}s train_nmse={row[1]:.5f} val_mae={row[2]:.5f}m val_euclid={row[3]:.5f}m")
        if met["position_MAE_m"]<best: best=met["position_MAE_m"]; torch.save(model.state_dict(),ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=device)); final=eval_object(model,vl,mean,std); final.update({"mode":"object","encoder":cfg.encoder_name,"finetuned_encoder_checkpoint":finetuned_encoder_checkpoint(cfg),"best_position_MAE_m":best}); json.dump(final,open(d/"results.json","w"),indent=2)


@hydra.main(config_path="../configs",config_name="idm_fdm_robocasa")
def main(cfg: DictConfig):
    seed_all(cfg.seed); device=torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.mode == "idm": run_idm(cfg,device)
    elif cfg.mode == "fdm": run_fdm(cfg,device)
    elif cfg.mode == "state": run_state(cfg,device)
    elif cfg.mode == "object": run_object(cfg,device)
    else: raise ValueError("mode must be idm, fdm, state, or object")
if __name__ == "__main__": main()
