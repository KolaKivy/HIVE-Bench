"""Black-box visual-feature contract shared by HIVE VLM adapters."""

from contextlib import nullcontext
from typing import Any, List, Optional

import torch
import torch.nn as nn


class VisualTokenEncoder(nn.Module):
    """Expose model-specific image representations as ``[B, N, H]`` tokens.

    The policy framework deliberately knows nothing about chat token ids, image
    layouts, processors, or the model's hidden-state return type. Adapters set
    ``image_token_id`` for placeholder-token architectures, or
    ``vision_token_count`` for fixed-prefix architectures such as OpenVLA.
    Models with different layouts can override ``encode_visual_tokens``.
    """

    image_token_id: Optional[int] = None
    vision_token_count: Optional[int] = None

    def encode_visual_tokens(
        self,
        images: List[List[Any]],
        instructions: List[str],
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Return padded visual tokens with shape ``[batch, tokens, hidden]``."""
        inputs = self.build_qwenvl_inputs(images=images, instructions=instructions)
        frozen = not any(parameter.requires_grad for parameter in self.parameters())
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            outputs = self(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = self._select_hidden_state(outputs, layer_idx)
        return self._extract_visual_tokens(hidden, inputs)

    @staticmethod
    def _select_hidden_state(outputs: Any, layer_idx: int) -> torch.Tensor:
        hidden_states = outputs.hidden_states
        if isinstance(hidden_states, torch.Tensor):
            if layer_idx not in (-1, None):
                raise ValueError(
                    "This VLM exposes only its final hidden state; set "
                    "framework.vision_text_fusion.vlm_layer_idx to -1."
                )
            return hidden_states
        if hidden_states is None:
            raise RuntimeError("VLM did not return hidden states required for visual features.")
        return hidden_states[layer_idx]

    def _extract_visual_tokens(self, hidden: torch.Tensor, inputs: Any) -> torch.Tensor:
        if self.vision_token_count is not None:
            return hidden[:, 1 : 1 + self.vision_token_count, :]

        if self.image_token_id is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set image_token_id, vision_token_count, "
                "or override _extract_visual_tokens()."
            )

        input_ids = inputs["input_ids"]
        batch_size, _, hidden_size = hidden.shape
        per_sample = [hidden[index][input_ids[index] == self.image_token_id] for index in range(batch_size)]
        if not per_sample or any(tokens.numel() == 0 for tokens in per_sample):
            counts = [tokens.shape[0] for tokens in per_sample]
            raise RuntimeError(
                f"{type(self).__name__} found no image placeholder tokens in one or more samples: {counts}."
            )

        max_tokens = max(tokens.shape[0] for tokens in per_sample)
        result = hidden.new_zeros(batch_size, max_tokens, hidden_size)
        for index, tokens in enumerate(per_sample):
            result[index, : tokens.shape[0]] = tokens
        return result
