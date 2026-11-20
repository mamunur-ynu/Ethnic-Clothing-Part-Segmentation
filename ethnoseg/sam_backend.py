"""SAM 3 through Hugging Face transformers (Sam3Model / Sam3Processor).

The image encoder runs once per image; every text prompt reuses those features
(model.get_vision_features -> model(vision_embeds=..., input_ids=...)), following
the transformers SAM 3 documentation. Runs on the Modal GPU (see modal_segment.py).
"""
from __future__ import annotations

import numpy as np


class Sam3Prompter:
    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None,
                 score_threshold: float = 0.3, mask_threshold: float = 0.5):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.score_threshold = score_threshold     # keep instances above this (rules apply stricter ones)
        self.mask_threshold = mask_threshold
        self._text_cache: dict[str, dict] = {}

    def _text(self, prompt: str):
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self.processor(text=prompt, return_tensors="pt").to(self.device)
        return self._text_cache[prompt]

    def __call__(self, image, prompts: list[str]) -> dict[str, list[tuple[np.ndarray, float]]]:
        """image: PIL RGB. Returns {prompt: [(bool mask HxW, score), ...]} at the image's size."""
        torch = self.torch
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        sizes = img_inputs["original_sizes"].tolist()
        out: dict[str, list[tuple[np.ndarray, float]]] = {}
        with torch.inference_mode():
            vision = self.model.get_vision_features(pixel_values=img_inputs["pixel_values"])
            for p in dict.fromkeys(prompts):           # unique, order kept
                outputs = self.model(vision_embeds=vision, **self._text(p))
                res = self.processor.post_process_instance_segmentation(
                    outputs, threshold=self.score_threshold, mask_threshold=self.mask_threshold,
                    target_sizes=sizes)[0]
                masks = res["masks"].bool().cpu().numpy()
                scores = res["scores"].float().cpu().numpy()
                out[p] = [(masks[i], float(scores[i])) for i in range(len(scores))]
        return out
