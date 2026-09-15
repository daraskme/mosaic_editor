"""SAM2.1 (Meta) による bbox → 輪郭マスクの再セグメント.

検出器が返した bbox を「箱の中のその物体」の輪郭マスクにする。
facebook/sam2.1-hiera-* は非gated・Apache-2.0 で、HF ログイン不要。
"""
from __future__ import annotations

import gc
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from .base import ProgressCB, pick_device, pick_dtype

logger = logging.getLogger(__name__)


class Sam2BoxRefiner:
    MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.model: Optional[Sam2Model] = None
        self.processor: Optional[Sam2Processor] = None

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        self.device = pick_device()
        self.dtype = pick_dtype(self.device)
        if progress_cb:
            progress_cb(f"SAM2.1 をロード中 (device={self.device})...\n"
                        "初回は ~900MB のダウンロードが発生します")
        # sam2.1 チェックポイントは model_type=sam2_video のため、Sam2Model への
        # ロードで無害な警告が出る。ロードの間だけ抑制する
        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            self.model = Sam2Model.from_pretrained(
                self.MODEL_ID, torch_dtype=self.dtype,
            ).to(self.device).eval()
        finally:
            transformers.logging.set_verbosity(prev)
        self.processor = Sam2Processor.from_pretrained(self.MODEL_ID)
        self._loaded = True

    def segment_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """bbox 内の物体の輪郭マスク (uint8 H×W, 0/255) を返す."""
        return self.segment_boxes(image, [box])[0]

    def segment_boxes(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
        progress_cb: ProgressCB = None,
    ) -> List[Optional[np.ndarray]]:
        """画像特徴を1回だけ計算し、各 bbox を順に輪郭マスク化する."""
        if not boxes:
            return []
        self.load(progress_cb)
        model, processor = self.model, self.processor
        if model is None or processor is None:
            raise RuntimeError("SAM2 のロードが完了していません")
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(
            images=image,
            input_boxes=[[list(map(float, box)) for box in boxes]],
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)
        masks: List[Optional[np.ndarray]] = []
        with torch.inference_mode():
            if progress_cb:
                progress_cb("SAM2 の画像特徴を計算中...")
            embeddings = model.get_image_embeddings(inputs["pixel_values"])
            for i, box in enumerate(boxes):
                if progress_cb:
                    progress_cb(f"SAM2 で輪郭マスク化 [{i + 1}/{len(boxes)}]...")
                try:
                    outputs = model(
                        image_embeddings=embeddings,
                        input_boxes=inputs["input_boxes"][:, i:i + 1],
                        multimask_output=True,
                    )
                    best = int(outputs.iou_scores[0, 0].argmax())
                    mask = processor.post_process_masks(
                        outputs.pred_masks[:, :, best:best + 1].float().cpu(),
                        inputs["original_sizes"],
                    )[0][0, 0]
                    masks.append(mask.numpy().astype(np.uint8) * 255)
                except Exception as exc:
                    logger.warning("SAM2 refinement failed for %s: %s", box, exc)
                    if progress_cb:
                        progress_cb(f"SAM2 輪郭化に失敗、矩形を使用: {exc}")
                    masks.append(None)
        return masks

    def unload(self):
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
