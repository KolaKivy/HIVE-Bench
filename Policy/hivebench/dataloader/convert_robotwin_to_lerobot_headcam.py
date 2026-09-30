'Module documentation.'

import argparse
import json
import os
import pickle
import random
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import io
import cv2
from tqdm import tqdm


CAM_MAP = {
    # HIVE-Bench official Robotwin convention: cam_high is the top-down head camera.
    "cam_high":        "observation/head_camera/rgb",
    "cam_left_wrist":  "observation/left_camera/rgb",
    "cam_right_wrist": "observation/right_camera/rgb",
}

MODALITY_JSON = {
    "action": {
        "left_joints":   {"start": 0,  "end": 6,  "original_key": "action"},
        "left_gripper":  {"start": 6,  "end": 7,  "original_key": "action"},
        "right_joints":  {"start": 7,  "end": 13, "original_key": "action"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
    },
    "state": {
        "left_joints":   {"start": 0,  "end": 6,  "original_key": "observation.state"},
        "left_gripper":  {"start": 6,  "end": 7,  "original_key": "observation.state"},
        "right_joints":  {"start": 7,  "end": 13, "original_key": "observation.state"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
    },
    "video": {
        "cam_high":       {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist":{"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"}
    },
}


def decode_jpeg_bytes(raw: bytes) -> np.ndarray:
    """JPEG bytes → RGB numpy array"""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_video(frames: list, out_path: Path, fps: int = 10):
    """frames: list of RGB numpy arrays → mp4"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def get_instruction(inst_dir: Path, ep_idx: int) -> str:
    'Get instruction function.'
    p = inst_dir / f"episode{ep_idx}.json"
    if not p.exists():
        return "robot manipulation task"
    with open(p) as f:
        data = json.load(f)
    seen = data.get("seen", [])
    if seen:
        return random.choice(seen)
    unseen = data.get("unseen", [])
    if unseen:
        return random.choice(unseen)
    return "robot manipulation task"


def convert_task(src_task_dir: Path, dst_task_dir: Path, split: str):
    'Convert task function.'
    
    split_subdir = "demo_randomized" if split.lower() == "randomized" else "demo_clean"
    demo_dir = src_task_dir / split_subdir

    if not demo_dir.exists():
        print(f"  [SKIP] {demo_dir} not found")
        return

    hdf5_files = sorted(demo_dir.glob("data/episode*.hdf5"),
                        key=lambda p: int(p.stem.replace("episode", "")))
    if not hdf5_files:
        print(f"  [SKIP] no hdf5 files in {demo_dir}/data/")
        return

    inst_dir = demo_dir / "instructions"
    dst_task_dir.mkdir(parents=True, exist_ok=True)

    
    all_rows = []        # parquet rows
    episodes_meta = []   # episodes.jsonl
    tasks_set = {}       # task_str → task_index
    global_step = 0

    video_frames = {cam: [] for cam in CAM_MAP}  # cam → list of (ep_idx, frames)

    print(f"  Converting {len(hdf5_files)} episodes ...")
    for ep_file in tqdm(hdf5_files, leave=False):
        ep_num = int(ep_file.stem.replace("episode", ""))

        with h5py.File(ep_file, "r") as f:
            action_vec = f["joint_action/vector"][:]          # [T, 14]
            state_vec  = f["joint_action/vector"][:]          # state = action (t)
            T = action_vec.shape[0]

            
            ep_frames = {}
            for cam_key, hdf5_key in CAM_MAP.items():
                raw_list = f[hdf5_key][:]   # [T,] bytes
                frames = [decode_jpeg_bytes(bytes(r)) for r in raw_list]
                ep_frames[cam_key] = frames

        
        task_str = get_instruction(inst_dir, ep_num)
        if task_str not in tasks_set:
            tasks_set[task_str] = len(tasks_set)
        task_idx = tasks_set[task_str]

        
        ep_idx = len(episodes_meta)

        # parquet rows
        for t in range(T):
            all_rows.append({
                "observation.state": state_vec[t].astype(np.float32).tolist(),
                "action":            action_vec[t].astype(np.float32).tolist(),
                "timestamp":         float(t) / 10.0,
                "frame_index":       t,
                "episode_index":     ep_idx,
                "index":             global_step,
                "task_index":        task_idx,
            })
            global_step += 1

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [task_str],
            "length": T,
        })

        
        for cam_key, frames in ep_frames.items():
            video_frames[cam_key].append((ep_idx, frames))

    
    data_dir = dst_task_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(all_rows)
    for ep_idx in range(len(episodes_meta)):
        ep_df = df_all[df_all["episode_index"] == ep_idx].copy()
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(parquet_path, index=False)
    print(f"  Wrote {len(episodes_meta)} parquet files → {data_dir}")

    
    for cam_key, ep_list in video_frames.items():
        cam_dir = dst_task_dir / "videos" / "chunk-000" / f"observation.images.{cam_key}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for ep_idx, frames in tqdm(ep_list, desc=f"  video/{cam_key}", leave=False):
            out_path = cam_dir / f"episode_{ep_idx:06d}.mp4"
            write_video(frames, out_path)

    
    meta_dir = dst_task_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # modality.json
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(MODALITY_JSON, f, indent=4)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_str, task_idx in sorted(tasks_set.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_str}) + "\n")

    # info.json
    n_ep = len(episodes_meta)
    joint_names = [
        "left_waist", "left_shoulder", "left_elbow",
        "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
        "left_gripper",
        "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
        "right_gripper",
    ]
    video_feature = {
        "dtype": "video",
        "shape": [3, 240, 320],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 240,
            "video.width": 320,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "robotwin50",
        "total_episodes": n_ep,
        "total_frames": global_step,
        "total_tasks": len(tasks_set),
        "total_videos": n_ep * 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [14], "names": [joint_names]
            },
            "action": {
                "dtype": "float32", "shape": [14], "names": [joint_names]
            },
            "observation.images.cam_high":        video_feature,
            "observation.images.cam_left_wrist":  video_feature,
            "observation.images.cam_right_wrist": video_feature,
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    
    action_arr = np.array([r["action"] for r in all_rows], dtype=np.float32)
    state_arr  = np.array([r["observation.state"] for r in all_rows], dtype=np.float32)
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for ep in episodes_meta:
            ep_rows = [r for r in all_rows if r["episode_index"] == ep["episode_index"]]
            a = np.array([r["action"] for r in ep_rows], dtype=np.float32)
            s = np.array([r["observation.state"] for r in ep_rows], dtype=np.float32)
            stats = {
                "episode_index": ep["episode_index"],
                "stats": {
                    "action": {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                               "min": a.min(0).tolist(), "max": a.max(0).tolist()},
                    "observation.state": {"mean": s.mean(0).tolist(), "std": s.std(0).tolist(),
                                         "min": s.min(0).tolist(), "max": s.max(0).tolist()},
                }
            }
            f.write(json.dumps(stats) + "\n")

    
    stats_gr00t = {
        "action": {
            "mean": action_arr.mean(0).tolist(),
            "std":  action_arr.std(0).tolist(),
            "min":  action_arr.min(0).tolist(),
            "max":  action_arr.max(0).tolist(),
            "q01":  np.quantile(action_arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(action_arr, 0.99, axis=0).tolist(),
        },
        "observation.state": {
            "mean": state_arr.mean(0).tolist(),
            "std":  state_arr.std(0).tolist(),
            "min":  state_arr.min(0).tolist(),
            "max":  state_arr.max(0).tolist(),
            "q01":  np.quantile(state_arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(state_arr, 0.99, axis=0).tolist(),
        },
    }
    with open(meta_dir / "stats_gr00t.json", "w") as f:
        json.dump(stats_gr00t, f, indent=4)

    # steps_data_index.pkl
    steps_index = []
    for ep in episodes_meta:
        start = sum(e["length"] for e in episodes_meta[:ep["episode_index"]])
        end   = start + ep["length"]
        steps_index.append((start, end))
    with open(meta_dir / "steps_data_index.pkl", "wb") as f:
        pickle.dump(steps_index, f)

    print(f"  Done: {n_ep} episodes, {global_step} steps, {len(tasks_set)} tasks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True,
                        help="Source dataset root, for example playground/Datasets/data")
    parser.add_argument("--dst_root", required=True,
                        help="Output root, for example playground/Datasets/RoboTwin_LeRobot")
    parser.add_argument("--split", default="Randomized",
                        choices=["Randomized", "Clean"],
                        help="Dataset split to convert")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task names to convert; omit to convert all tasks")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if args.tasks:
        task_names = args.tasks
    else:
        task_names = sorted([p.name for p in src_root.iterdir() if p.is_dir()])

    print(f"Converting {len(task_names)} tasks: {task_names}")
    print(f"  src: {src_root}")
    print(f"  dst: {dst_root / args.split}")

    for task_name in task_names:
        src_task = src_root / task_name
        dst_task = dst_root / args.split / task_name
        print(f"\n[{task_name}]")
        convert_task(src_task, dst_task, args.split)

    print("\nAll done!")


if __name__ == "__main__":
    main()