"""16-step, policy-aligned RoboCasa inverse-dynamics dataset."""
import glob
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Exact order used by FourierGr1ArmsWaistDataConfig: LA, RA, LH, RH, waist.
CONTROL_ACTION_INDICES = tuple(list(range(0, 7)) + list(range(22, 29)) + list(range(7, 13)) + list(range(29, 35)) + list(range(41, 44)))
CONTROL_GROUPS = {"left_arm": list(range(0, 7)), "right_arm": list(range(7, 14)), "left_hand": list(range(14, 20)), "right_hand": list(range(20, 26)), "waist": list(range(26, 29))}

# RoboTwin converted LeRobot action: left arm (6), left gripper (1), right arm (6), right gripper (1).
ROBOTWIN_ACTION_INDICES = tuple(range(14))
ROBOTWIN_CONTROL_GROUPS = {"left_arm": list(range(0, 6)), "left_gripper": [6], "right_arm": list(range(7, 13)), "right_gripper": [13]}
def _video(path):
    try:
        import av
        c = av.open(path)
        out = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        return out
    except ImportError:
        import cv2
        cap, out = cv2.VideoCapture(path), []
        while True:
            ok, frame = cap.read()
            if not ok: break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return out


def _alloc(total, names, rng):
    q, r = divmod(total, len(names)); out = {n: q for n in names}
    order = list(names); rng.shuffle(order)
    for n in order[:r]: out[n] += 1
    return out


class _Episode:
    def __init__(self, parquet, video, horizon, indices):
        d = pd.read_parquet(parquet, columns=["action", "observation.state"])
        self.a = np.asarray(d["action"].tolist(), np.float32)
        self.s = np.asarray(d["observation.state"].tolist(), np.float32)
        self.f, self.h, self.i = _video(video), horizon, np.asarray(indices)
        self.n = min(len(self.a), len(self.s), len(self.f))
    def __len__(self): return max(0, self.n - self.h)
    def __getitem__(self, t):
        # Sum of policy deltas a[t]-s[t], a[t+1]-a[t], ..., a[t+h-1]-a[t+h-2].
        y = self.a[t + self.h - 1, self.i] - self.s[t, self.i]
        return Image.fromarray(self.f[t]), Image.fromarray(self.f[t + self.h]), y.astype(np.float32)


class IDMDataset(Dataset):
    """Episode-stratified set; target is a[t+horizon-1] - state[t]."""
    def __init__(self, data_root, dataset_names, split, camera_key="observation.images.ego_view", horizon=16,
                 train_episodes=200, val_episodes=100, seed=42, action_indices=CONTROL_ACTION_INDICES,
                 normalization=None):
        if split not in {"train", "val"}: raise ValueError(split)
        self.root, self.names, self.split = Path(data_root), tuple(dataset_names), split
        self.camera, self.horizon, self.indices = camera_key, int(horizon), tuple(action_indices)
        if self.horizon < 1 or not self.names: raise ValueError("positive horizon and nonempty dataset_names required")
        rng = random.Random(seed)
        ntr, nva = _alloc(train_episodes, self.names, rng), _alloc(val_episodes, self.names, rng)
        self.pairs = []
        for name in self.names:
            root = self.root / name
            ps = []
            for x in sorted(glob.glob(str(root / "data" / "chunk-*" / "episode_*.parquet"))):
                q = Path(x); v = root / "videos" / q.parent.name / self.camera / (q.stem + ".mp4")
                if v.exists(): ps.append((name, str(q), str(v)))
            rng.shuffle(ps); need = ntr[name] + nva[name]
            if len(ps) < need: raise ValueError(f"{name}: need {need}, found {len(ps)} paired episodes")
            if split == "train": self.pairs += ps[:ntr[name]]
            else: self.pairs += ps[ntr[name]:need]
        self.episodes = [None] * len(self.pairs)
        self.flat = []
        for ei, (_, pq, _) in enumerate(self.pairs):
            n = len(pd.read_parquet(pq, columns=["index"]))
            self.flat += [(ei, t) for t in range(max(0, n - self.horizon))]
        self.mean = self.std = None
        if normalization: self.set_normalization(normalization)
        print(f"[IDMDataset] {split}: episodes={len(self.pairs)}, samples={len(self.flat)}, h={self.horizon}, tasks={dict(Counter(x[0] for x in self.pairs))}")

    def stats(self, eps=1e-6):
        n, s1, s2, ix = 0, np.zeros(len(self.indices), np.float64), np.zeros(len(self.indices), np.float64), np.asarray(self.indices)
        for _, pq, _ in self.pairs:
            d = pd.read_parquet(pq, columns=["action", "observation.state"])
            a, st = np.asarray(d["action"].tolist(), np.float64), np.asarray(d["observation.state"].tolist(), np.float64)
            m = min(len(a), len(st)) - self.horizon
            if m > 0:
                y = a[self.horizon-1:m+self.horizon-1, ix] - st[:m, ix]
                n += m; s1 += y.sum(0); s2 += (y*y).sum(0)
        if not n: raise ValueError("No valid target windows")
        mean = s1/n; std = np.maximum(np.sqrt(np.maximum(s2/n - mean*mean, 0)), eps)
        return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist(), "count": n}

    def set_normalization(self, x):
        self.mean, self.std = np.asarray(x["mean"], np.float32), np.maximum(np.asarray(x["std"], np.float32), 1e-6)
        if self.mean.shape != (len(self.indices),) or self.std.shape != self.mean.shape: raise ValueError("bad target normalization shape")
    def manifest(self): return [{"task": a, "parquet": b, "video": c} for a,b,c in self.pairs]
    def __len__(self): return len(self.flat)
    def __getitem__(self, k):
        ei, t = self.flat[k]
        if self.episodes[ei] is None:
            _, pq, v = self.pairs[ei]; self.episodes[ei] = _Episode(pq, v, self.horizon, self.indices)
        x, xh, y = self.episodes[ei][t]
        if self.mean is not None: y = (y - self.mean) / self.std
        return x, xh, torch.from_numpy(y), self.pairs[ei][0]


def collate_fn(batch):
    return [x[0] for x in batch], [x[1] for x in batch], torch.stack([x[2] for x in batch]), [x[3] for x in batch]


class FDMDataset(IDMDataset):
    """Same sampled trajectories as IDM, but yields 16 normalized policy-delta tokens."""
    def __init__(self, *args, action_normalization=None, **kwargs):
        super().__init__(*args, normalization=None, **kwargs)
        self.action_mean = self.action_std = None
        if action_normalization is not None:
            self.set_action_normalization(action_normalization)

    def action_stats(self, eps=1e-6):
        """Train-only statistics over every individual delta token in valid windows."""
        count = 0
        total = np.zeros(len(self.indices), np.float64)
        total_sq = np.zeros(len(self.indices), np.float64)
        ix = np.asarray(self.indices)
        for _, pq, _ in self.pairs:
            d = pd.read_parquet(pq, columns=["action", "observation.state"])
            a = np.asarray(d["action"].tolist(), np.float64)
            st = np.asarray(d["observation.state"].tolist(), np.float64)
            m = min(len(a), len(st)) - self.horizon
            if m <= 0: continue
            first = a[:m, ix] - st[:m, ix]
            delta = a[1:, ix] - a[:-1, ix]
            future = np.stack([delta[j:j+m] for j in range(self.horizon - 1)], axis=1)
            seq = np.concatenate([first[:, None], future], axis=1)
            count += seq.shape[0] * seq.shape[1]
            total += seq.sum(axis=(0, 1)); total_sq += (seq * seq).sum(axis=(0, 1))
        if not count: raise ValueError("No valid FDM action tokens")
        mean = total / count; std = np.maximum(np.sqrt(np.maximum(total_sq / count - mean * mean, 0)), eps)
        return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist(), "count": count}

    def set_action_normalization(self, x):
        self.action_mean = np.asarray(x["mean"], np.float32)
        self.action_std = np.maximum(np.asarray(x["std"], np.float32), 1e-6)
        if self.action_mean.shape != (len(self.indices),) or self.action_std.shape != self.action_mean.shape:
            raise ValueError("bad FDM action-normalization shape")

    def __getitem__(self, k):
        ei, t = self.flat[k]
        if self.episodes[ei] is None:
            _, pq, v = self.pairs[ei]; self.episodes[ei] = _Episode(pq, v, self.horizon, self.indices)
        ep = self.episodes[ei]
        ix = np.asarray(self.indices)
        first = ep.a[t, ix] - ep.s[t, ix]
        future = ep.a[t+1:t+self.horizon, ix] - ep.a[t:t+self.horizon-1, ix]
        actions = np.concatenate([first[None], future], axis=0).astype(np.float32)
        if self.action_mean is not None: actions = (actions - self.action_mean) / self.action_std
        return (Image.fromarray(ep.f[t]), Image.fromarray(ep.f[t+self.horizon]),
                torch.from_numpy(actions), self.pairs[ei][0])


def fdm_collate_fn(batch):
    return [x[0] for x in batch], [x[1] for x in batch], torch.stack([x[2] for x in batch]), [x[3] for x in batch]


class StateDataset(IDMDataset):
    """Single-frame visual state probe using the exact IDM/FDM episode split."""
    def __init__(self, *args, state_normalization=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("normalization", None)
        super().__init__(*args, normalization=None, **kwargs)
        # IDM/FDM require a future frame and therefore omit the final horizon
        # frames. State has no future dependency, so retain every frame from the
        # same selected episodes.
        self.flat = []
        for ei, (_, pq, _) in enumerate(self.pairs):
            n = len(pd.read_parquet(pq, columns=["index"]))
            self.flat += [(ei, t) for t in range(n)]
        self.state_mean = self.state_std = None
        if state_normalization is not None: self.set_state_normalization(state_normalization)
        print(f"[StateDataset] {self.split}: episodes={len(self.pairs)}, samples={len(self.flat)}")

    def state_stats(self, eps=1e-6):
        n=0; total=np.zeros(len(self.indices),np.float64); sq=np.zeros(len(self.indices),np.float64); ix=np.asarray(self.indices)
        for _, pq, _ in self.pairs:
            d=pd.read_parquet(pq, columns=["observation.state"]); x=np.asarray(d["observation.state"].tolist(),np.float64)[:,ix]
            n += len(x); total += x.sum(0); sq += (x*x).sum(0)
        mean=total/n; std=np.maximum(np.sqrt(np.maximum(sq/n-mean*mean,0)),eps)
        return {"mean":mean.astype(np.float32).tolist(),"std":std.astype(np.float32).tolist(),"count":n}

    def set_state_normalization(self, x):
        self.state_mean=np.asarray(x["mean"],np.float32); self.state_std=np.maximum(np.asarray(x["std"],np.float32),1e-6)
        if self.state_mean.shape != (len(self.indices),) or self.state_std.shape != self.state_mean.shape: raise ValueError("bad state-normalization shape")

    def __getitem__(self, k):
        ei,t=self.flat[k]
        if self.episodes[ei] is None:
            _,pq,v=self.pairs[ei]; self.episodes[ei]=_Episode(pq,v,self.horizon,self.indices)
        ep=self.episodes[ei]; target=ep.s[t,np.asarray(self.indices)].astype(np.float32)
        if self.state_mean is not None: target=(target-self.state_mean)/self.state_std
        return Image.fromarray(ep.f[t]), torch.from_numpy(target), self.pairs[ei][0]


def state_collate_fn(batch):
    return [x[0] for x in batch], torch.stack([x[1] for x in batch]), [x[2] for x in batch]


# One movable/visually-localized task object per selected task.
OBJECT_TASKS = {
    "adjust_bottle": ("001_bottle",),
    "beat_block_hammer": ("box",),
    "click_alarmclock": ("046_alarm-clock",),
    "open_laptop": ("015_laptop",),
}
OBJECT_NUM_QUERIES = 1


class RoboTwinObjectDataset(Dataset):
    """Initial head-camera frame -> task-object camera-frame positions."""
    def __init__(self, data_root, split, train_per_task=400, val_per_task=100, seed=42, normalization=None):
        if split not in {"train", "val"}: raise ValueError(split)
        self.root=Path(data_root); self.split=split; self.records=[]
        for task, objects in OBJECT_TASKS.items():
            base=self.root/task/'demo_randomized'; scene=json.load(open(base/'scene_info.json'))
            available=[]
            for key, info in scene.items():
                try: idx=int(key.rsplit('_',1)[1])
                except (IndexError,ValueError): continue
                h5=base/'data'/f'episode{idx}.hdf5'
                if h5.exists() and all(o is None or o in info['object_poses'] for o in objects): available.append((idx,info,h5))
            rng=random.Random(seed + sum(map(ord,task))); rng.shuffle(available)
            need=train_per_task+val_per_task
            if len(available)<need: raise ValueError(f"{task}: need {need} valid episodes, found {len(available)}")
            selected=available[:train_per_task] if split=='train' else available[train_per_task:need]
            for idx,info,h5 in selected:
                target=np.zeros((OBJECT_NUM_QUERIES,3),np.float32); mask=np.zeros(OBJECT_NUM_QUERIES,np.float32)
                for slot,name in enumerate(objects):
                    if name is not None:
                        target[slot]=np.asarray(info['object_poses'][name]['position_camera'],np.float32); mask[slot]=1.
                self.records.append({"task":task,"episode":idx,"h5":str(h5),"target":target,"mask":mask})
        self.mean=self.std=None
        if normalization is not None: self.set_normalization(normalization)
        print(f"[RoboTwinObjectDataset] {split}: episodes={len(self.records)}, tasks={dict(Counter(x['task'] for x in self.records))}")

    def stats(self, eps=1e-6):
        vals=np.concatenate([x['target'][x['mask'].astype(bool)] for x in self.records],axis=0)
        return {"mean":vals.mean(0).astype(np.float32).tolist(),"std":np.maximum(vals.std(0),eps).astype(np.float32).tolist(),"count":len(vals)}

    def set_normalization(self,x):
        self.mean=np.asarray(x['mean'],np.float32); self.std=np.maximum(np.asarray(x['std'],np.float32),1e-6)
        if self.mean.shape!=(3,) or self.std.shape!=(3,): raise ValueError("object position normalization must be 3-D")

    def manifest(self): return [{"task":x['task'],"episode":x['episode'],"h5":x['h5'],"objects":OBJECT_TASKS[x['task']]} for x in self.records]
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        import h5py
        r=self.records[i]
        with h5py.File(r['h5'],'r') as f: image=Image.open(io.BytesIO(bytes(f['observation/head_camera/rgb'][0]))).convert('RGB')
        target=r['target'].copy()
        if self.mean is not None: target=(target-self.mean)/self.std
        return image,torch.from_numpy(target),torch.from_numpy(r['mask']),r['task']


def object_collate_fn(batch):
    return [x[0] for x in batch],torch.stack([x[1] for x in batch]),torch.stack([x[2] for x in batch]),[x[3] for x in batch]
