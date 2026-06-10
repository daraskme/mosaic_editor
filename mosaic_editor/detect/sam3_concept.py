"""SAM3 (Meta) Promptable Concept Segmentation バックエンド.

テキストプロンプト (短い名詞句) から、一致する全インスタンスの
輪郭マスク + bbox + スコアを直接得る。検出とセグメントが1モデルで完結。

- HF: facebook/sam3 (gated — 初回は HF ログインと利用許諾が必要)
- transformers >= 5.0.0
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from ..core.categories import Category
from .base import Detection, ProgressCB, pick_device, pick_dtype


class Sam3ConceptSegmenter:
    """Sam3Model による text → instance masks.

    facebook/sam3 (gated) にアクセスできない場合は、コミュニティ版の
    SAM3 Lite Text (テキストエンコーダのみ軽量化した互換チェックポイント)
    に自動フォールバックする。
    """

    MODEL_ID = "facebook/sam3"
    FALLBACK_MODEL_ID = "yonigozlan/sam3-litetext-s0"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.model = None
        self.processor = None
        self.active_model_id: Optional[str] = None

    def _official_accessible(self) -> bool:
        from huggingface_hub import hf_hub_download
        try:
            hf_hub_download(self.MODEL_ID, "config.json")
            return True
        except Exception:
            return False

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        from transformers import Sam3LiteTextModel, Sam3Model, Sam3Processor

        self.device = pick_device()
        self.dtype = pick_dtype(self.device)

        if self._official_accessible():
            self.active_model_id = self.MODEL_ID
            model_cls = Sam3Model
        else:
            self.active_model_id = self.FALLBACK_MODEL_ID
            model_cls = Sam3LiteTextModel
            if progress_cb:
                progress_cb("facebook/sam3 (gated) にアクセスできないため\n"
                            "SAM3 Lite Text にフォールバックします")

        if progress_cb:
            progress_cb(f"SAM3 をロード中 ({self.active_model_id}, "
                        f"device={self.device})...\n"
                        "初回はモデルのダウンロードが発生します")
        self.model = model_cls.from_pretrained(
            self.active_model_id, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(self.active_model_id)
        self._loaded = True

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        """カテゴリごとの各プロンプトで PCS を実行し、検出を集める."""
        import torch

        self.load(progress_cb=progress_cb)
        if image.mode != "RGB":
            image = image.convert("RGB")

        detections: List[Detection] = []
        n_total = sum(len(c.prompts) for c in categories)
        done = 0
        for cat in categories:
            for prompt in cat.prompts:
                done += 1
                if progress_cb:
                    progress_cb(f"SAM3 検出中 [{done}/{n_total}] 「{prompt}」...")
                inputs = self.processor(
                    images=image, text=prompt, return_tensors="pt"
                ).to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                for mask_t, box_t, score_t in zip(
                    results["masks"], results["boxes"], results["scores"]
                ):
                    mask = (mask_t.cpu().numpy() > 0).astype(np.uint8) * 255
                    x1, y1, x2, y2 = (int(round(float(v))) for v in box_t.tolist())
                    detections.append(Detection(
                        label=cat.label,
                        category_key=cat.key,
                        bbox=(x1, y1, x2, y2),
                        score=float(score_t),
                        mask=mask,
                    ))
        return detections

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
