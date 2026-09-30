import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer

from hivebench.model.framework.base_framework import baseframework
from hivebench.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from hivebench.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register('Dinov3CLIPGR00T')
class Dinov3CLIPGR00T(baseframework):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vision_model_name = config.framework.get('vision_model', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
        self.clip_model_name = config.framework.get('clip_model', config.framework.get('text_model', 'openai/clip-vit-base-patch32'))

        # Reuse DinoGR00T's encoder factory so every supported vision backbone
        # shares its own preprocessing and returns [B, N, D] patch tokens.
        # Keep the existing full DINOv3 model id working with its register-token
        # aware wrapper; all other names/IDs follow DinoGR00T exactly.
        from hivebench.model.framework.DinoGR00T import _build_vision_encoder
        encoder_name = self.vision_model_name
        if encoder_name == 'facebook/dinov3-vitb16-pretrain-lvd1689m':
            encoder_name = 'dinov3'
        print(f"Initializing vision backbone: {self.vision_model_name}")
        self.vision_encoder = _build_vision_encoder(encoder_name)

        print(f"Initializing CLIP text encoder: {self.clip_model_name}")
        self.text_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(self.clip_model_name)

        self.vision_dim = self.vision_encoder.num_channels
        self.text_dim = self.text_encoder.config.hidden_size
        self.text_projector = nn.Identity() if self.text_dim == self.vision_dim else nn.Linear(self.text_dim, self.vision_dim)

        self.config.framework.action_model.hidden_size = self.vision_dim
        if hasattr(self.config.framework.action_model, 'diffusion_model_cfg'):
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.vision_dim

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def _extract_patch_tokens(self, image_tensors: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image_tensors)

    def _encode_images(self, batch_images: List[List[Image.Image]]) -> torch.Tensor:
        B = len(batch_images)
        num_view = len(batch_images[0])
        image_tensors = self.vision_encoder.prepare_dino_input(batch_images)
        patch_tokens = self._extract_patch_tokens(image_tensors)
        _, num_patches, _ = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(B, num_view, num_patches, self.vision_dim)
        image_hidden = patch_tokens.reshape(B, num_view * num_patches, self.vision_dim)
        return image_hidden

    def _encode_text(self, instructions: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        text_inputs = self.text_tokenizer(
            instructions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device=device) for key, value in text_inputs.items()}
        text_outputs = self.text_encoder(**text_inputs)
        text_hidden = self.text_projector(text_outputs.last_hidden_state)
        text_hidden = text_hidden.to(dtype=dtype)
        return text_hidden

    def _build_last_hidden(self, batch_images: List[List[Image.Image]], instructions: List[str]) -> torch.Tensor:
        image_hidden = self._encode_images(batch_images)
        text_hidden = self._encode_text(instructions, image_hidden.device, image_hidden.dtype)
        last_hidden = torch.cat([image_hidden, text_hidden], dim=1)
        return last_hidden

    def _prepare_predict_images(self, batch_images: List[List[Image.Image]]) -> List[List[Image.Image]]:
        processed_batch_images = []
        for images in batch_images:
            processed_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    processed_images.append(Image.fromarray(img))
                else:
                    processed_images.append(img)
            processed_batch_images.append(processed_images)
        return processed_batch_images

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        last_hidden = self._build_last_hidden(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = getattr(self.config.framework.action_model, "repeated_diffusion_steps", 4)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> Dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        batch_images = self._prepare_predict_images(batch_images)
        last_hidden = self._build_last_hidden(batch_images, instructions)
        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)

        if isinstance(pred_actions, torch.Tensor):
            actions_out = pred_actions.detach().cpu().float().numpy()
        else:
            actions_out = pred_actions[0].detach().cpu().float().numpy() if isinstance(pred_actions, (list, tuple)) else pred_actions

        return {"normalized_actions": actions_out}
