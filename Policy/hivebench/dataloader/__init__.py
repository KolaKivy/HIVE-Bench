import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
# from hivebench.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



#     if dataset_py == "lerobot_datasets":
#         from hivebench.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
#         vla_dataset_cfg = cfg.datasets.vla_data

#         vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
#         vla_train_dataloader = DataLoader(
#             vla_dataset,
#             batch_size=cfg.datasets.vla_data.per_device_batch_size,
#             collate_fn=collate_fn,
#             num_workers=4,
#             # shuffle=True
#         )        
#         if dist.get_rank() == 0: 
            
#             output_dir = Path(cfg.output_dir)
#             vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
#         return vla_train_dataloader
#     elif dataset_py == "vlm_datasets":
#         vlm_data_module = make_vlm_dataloader(cfg)
#         vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
#         return vlm_train_dataloader
def build_dataloader(cfg, dataset_py="lerobot_datasets"):
    'Build dataloader function.'
    if dataset_py == "lerobot_datasets":
        from .lerobot_datasets import get_vla_dataset, collate_fn
        
        
        vla_dataset_cfg = cfg.datasets.vla_data
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        
        if dist.is_initialized():
            if dist.get_rank() == 0:
                output_dir = Path(cfg.run_root_dir) / cfg.run_id
                output_dir.mkdir(parents=True, exist_ok=True)
                stats_path = output_dir / "dataset_statistics.json"
                print(f"[INFO] Extracting and saving dataset statistics to: {stats_path}")
                try:
                    
                    vla_dataset.save_dataset_statistics(stats_path)
                except Exception as e:
                    print(f"Warning: Failed to save stats: {e}")
        
        
        from torch.utils.data import DataLoader
        vla_dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn
        )
        return vla_dataloader
    else:
        raise ValueError(f"Unknown dataset_py: {dataset_py}")
