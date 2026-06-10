"""SAM2.1 (Meta) による bbox → 輪郭マスクの再セグメント.

LocateAnything-3B が返した bbox を「箱の中のその物体」の輪郭マスクにする。
facebook/sam2.1-hiera-* は非gated・Apache-2.0 で、HF ログイン不要。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .base import ProgressCB, pick_device, pick_dtype


class Sam2BoxRefiner:
    MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.model = None
        self.processor = None

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        from transformers import Sam2Model, Sam2Processor

        self.device = pick_device()
        self.dtype = pick_dtype(self.device)
        if progress_cb:
            progress_cb(f"SAM2.1 をロード中 (device={self.device})...\n"
                        "初回は ~900MB のダウンロードが発生します")
        self.model = Sam2Model.from_pretrained(
            self.MODEL_ID, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.processor = Sam2Processor.from_pretrained(self.MODEL_ID)
        self._loaded = True

    def segment_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """bbox 内の物体の輪郭マスク (uint8 H×W, 0/255) を返す."""
        import torch

        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")

        x1, y1, x2, y2 = box
        inputs = self.processor(
            images=image,
            input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=True)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"]
        )[0]
        # masks: (num_boxes=1, num_multimask, H, W) — IoU 最良のマスクを採用
        iou = outputs.iou_scores.float().cpu()[0, 0]
        best = int(iou.argmax())
        mask = masks[0, best].numpy().astype(bool)
        return mask.astype(np.uint8) * 255

    def unload(self):
        import gc
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
