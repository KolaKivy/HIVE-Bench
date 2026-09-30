import collections
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from collections import deque

import numpy as np
import cv2 as cv
import json_numpy


try:
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
except ModuleNotFoundError:
    # HIVE-Bench keeps deployment/ below Policy/; this fallback lets RoboTwin
    # import the interface when only the repository root is on PYTHONPATH.
    from Policy.deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
# HIVE-Bench checkpoint/config helper.
from hivebench.model.tools import read_mode_config

try:
    from Bench.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 15,
        adaptive_ensemble_alpha=0.5,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        max_joint_delta: Optional[float] = None,
        use_state: bool = False,
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"max_joint_delta: {max_joint_delta}, use_state: {use_state} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode
        self.use_state = bool(use_state)
        self.max_joint_delta = None if max_joint_delta is None else float(max_joint_delta)
        if self.max_joint_delta is not None and self.max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive or null")
        self.action_log = []   
        self.unconstrained_action_log = []
        self.state_log = []    
        self.infer_steps = []
        self.episode_count = 0
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_logs")
        self.log_path = os.path.join(self.log_dir, "action_log.txt")
        os.makedirs(self.log_dir, exist_ok=True)
        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None

    def reset(self, task_description: str) -> None:
        if len(self.action_log) > 0:
            self._plot_and_save()
            self.action_log = []
            self.unconstrained_action_log = []
            self.state_log = []
            self.infer_steps = []
            self.episode_count += 1
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None
    def _plot_and_save(self):
        actions = np.array(self.action_log)   # [T, 14]
        T = actions.shape[0]
        joint_names = [
            "l_waist", "l_shoulder", "l_elbow", "l_forearm",
            "l_wrist_a", "l_wrist_r", "l_gripper",
            "r_waist", "r_shoulder", "r_elbow", "r_forearm",
            "r_wrist_a", "r_wrist_r", "r_gripper",
        ]
        fig, axes = plt.subplots(7, 2, figsize=(16, 20))
        axes = axes.flatten()
        for i in range(14):
            axes[i].plot(actions[:, i], label="action")
            if len(self.unconstrained_action_log) == T:
                unconstrained_actions = np.array(self.unconstrained_action_log)
                axes[i].plot(
                    unconstrained_actions[:, i],
                    label="model_target",
                    linestyle=":",
                    alpha=0.7,
                )
            if len(self.state_log) == T:
                states = np.array(self.state_log)
                axes[i].plot(states[:, i], label="state", linestyle="--", alpha=0.7)
            
            for s in self.infer_steps:
                axes[i].axvline(x=s, color='r', alpha=0.2, linewidth=0.5)
            axes[i].set_title(joint_names[i])
            axes[i].legend(fontsize=6)
            axes[i].grid(True)
        plt.suptitle(f"Episode {self.episode_count} — {T} steps", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, f"episode_{self.episode_count:03d}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"[action_log] Saved → {save_path}")

    def _limit_joint_delta(
        self,
        action: np.ndarray,
        state: Optional[np.ndarray],
    ) -> np.ndarray:
        """Limit only arm-joint motion in RoboTwin's [L6, Lg, R6, Rg] order."""
        action = np.asarray(action, dtype=np.float32).copy()
        if self.max_joint_delta is None or state is None:
            return action

        state = np.asarray(state, dtype=np.float32)
        if action.shape != (14,) or state.shape != (14,):
            raise ValueError(
                "RobotWin qpos guard requires 14-D [left6, left_gripper, right6, right_gripper] "
                f"vectors, got action={action.shape}, state={state.shape}"
            )

        arm_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        delta = action[arm_indices] - state[arm_indices]
        action[arm_indices] = state[arm_indices] + np.clip(
            delta,
            -self.max_joint_delta,
            self.max_joint_delta,
        )
        return action

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # The current RoboTwin checkpoint was trained with include_state=false
        # (the loader omits the state key).  Keep the default consistent with
        # training; state-conditioned checkpoints can opt in explicitly.
        model_state = (
            self._prepare_model_state(state)
            if self.use_state and state is not None
            else None
        )

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()
        # The action head was trained with a normalized state in
        # [left6, right6, left_gripper, right_gripper] order.  RoboTwin's
        # online state is [left6, left_gripper, right6, right_gripper].
        # Keep the online state for the simulator, but send the correctly
        # reordered/normalized state to the policy server.
        if model_state is not None:
            example_copy["state"] = model_state
        else:
            example_copy.pop("state", None)
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        
        if self.action_ensemble or step % action_chunk_size == 0 or self.raw_actions is None:
            self.infer_steps.append(step)
            with open(self.log_path, 'a') as f:
                f.write(
                    f"step={step}, chunk_size={action_chunk_size}, "
                    f"reason={'ensemble' if self.action_ensemble else ('no_cache' if self.raw_actions is None else f'step%chunk={step%action_chunk_size}')}, "
                    f"policy_state={np.round(model_state, 3).tolist() if model_state is not None else None}\n"
                )
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            # Unnormalize to get delta/rel values
            raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                normalization_mode=self.normalization_mode,
            )

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        
        if self.action_ensemble:
            
            current_action = self.action_ensembler.ensemble_action(self.raw_actions)
            
        else:
            
            action_idx = step % action_chunk_size
            if action_idx >= len(self.raw_actions):
                
                action_idx = len(self.raw_actions) - 1
            current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        unconstrained_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        current_action = self._limit_joint_delta(unconstrained_action, state)
        self.action_log.append(current_action.copy())
        self.unconstrained_action_log.append(unconstrained_action.copy())
        if state is not None:
            self.state_log.append(np.array(state).copy())
        with open(self.log_path, 'a') as f:
            f.write(
                f"step={step}, chunk_size={action_chunk_size}, "
                f"state={np.round(np.array(state), 3).tolist() if state is not None else None}, "
                f"model_target={np.round(unconstrained_action, 3).tolist()}, "
                f"executed_action={np.round(current_action, 3).tolist()}\n"
            )
        return current_action

    def _prepare_model_state(self, robotwin_state: np.ndarray) -> np.ndarray:
        """Convert RoboTwin qpos to the state representation used in training.

        RoboTwin: [left6, left_gripper, right6, right_gripper]
        Training: [left6, right6, left_gripper, right_gripper], min--max/binary
        """
        robotwin_state = np.asarray(robotwin_state, dtype=np.float32)
        if robotwin_state.shape != (14,):
            raise ValueError(
                "Expected RoboTwin state with shape (14,) in "
                "[left6, left_gripper, right6, right_gripper] order, got "
                f"{robotwin_state.shape}"
            )

        train_order_state = robotwin_state[
            [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
        ]
        return self.normalize_state(
            train_order_state,
            self.state_norm_stats,
            normalization_mode=self.normalization_mode,
        )

    @staticmethod
    def normalize_state(
        state: dict[str, np.ndarray],
        state_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        continuous_mask = [True, True, True, True, True, True, True, True, True, True, True, True, False, False]
        continuous_mask = np.array(continuous_mask, dtype=bool)
        state_high, state_low = ModelClient._get_normalization_bounds(
            state_norm_stats, normalization_mode=normalization_mode
        )
        valid_mask = continuous_mask & (state_high != state_low)
        normalized_state = np.where(
            valid_mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(
            ~continuous_mask,
            (normalized_state > 0.5).astype(normalized_state.dtype),
            normalized_state,
        )
        return normalized_state

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(
            action_norm_stats, normalization_mode=normalization_mode
        )
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        Convert delta actions to absolute actions.

        Training: delta[0] = a[0] - s[0], delta[t] = a[t] - a[t-1]
        Deployment: a[0] = delta[0] + base, a[t] = delta[t] + a[t-1]

        Where base is:
        - First chunk: initial_state (s_0)
        - Subsequent chunks: prev_action (last action from previous chunk)
        """
        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))

        # Determine base action
        base = self.prev_action if self.prev_action is not None else self.initial_state

        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]

        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """
        Convert relative actions to absolute actions.

        Training: rel[t] = a[t] - s[0]
        Deployment: a[t] = rel[t] + s[0]
        """
        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))

        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])

        return abs_actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)

        stats = norm_stats[unnorm_key]

        # Support two formats:
        # New format: {"robotwin": {"abs": {...}, "delta": {...}, "rel": {...}}}
        # Old format: {"robotwin": {"action": {...}, "state": {...}}}

        if action_mode in stats:
            # New format: directly use the corresponding mode stats
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            # Old format: only supports abs mode
            if action_mode != "abs":
                raise ValueError(
                    f"Statistics key `{unnorm_key}` only provides `abs` action stats, "
                    f"but action_mode=`{action_mode}` was requested."
                )
            return stats["action"]
        raise ValueError(
            f"Invalid statistics file format for key `{unnorm_key}`. "
            f"Available top-level keys: {sorted(stats.keys())}"
        )

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        available_keys = sorted(norm_stats.keys())
        if unnorm_key is None:
            if len(available_keys) == 1:
                return available_keys[0]
            raise ValueError(
                "`unnorm_key` must be provided when multiple normalization statistics are available. "
                f"Available keys: {available_keys}"
            )

        if unnorm_key not in norm_stats:
            # HIVE-Bench's RoboTwin datasets are tagged ``new_embodiment``;
            # older deploy_policy.yml files called the same block ``robotwin``.
            aliases = {"robotwin": "new_embodiment", "new_embodiment": "robotwin"}
            alias = aliases.get(unnorm_key)
            if alias in norm_stats:
                return alias
            raise KeyError(
                f"Unknown `unnorm_key`: `{unnorm_key}`. Available keys: {available_keys}"
            )

        return unnorm_key

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError(
                    "Normalization mode `q99` requires statistics keys `q01` and `q99`."
                )
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError(
                    "Normalization mode `min_max` requires statistics keys `min` and `max`."
                )
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(
            f"Unsupported normalization_mode: {normalization_mode}. Expected one of ['min_max', 'q99']."
        )


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )
    max_joint_delta = usr_args.get("max_joint_delta", None)
    use_state = usr_args.get("use_state", False)

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        max_joint_delta=max_joint_delta,
        use_state=use_state,
    )

def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    # HIVE-Bench official RoboTwin order: [head, left wrist, right wrist].
    # This matches convert_robotwin_to_lerobot_headcam.py (head_camera -> cam_high).
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
