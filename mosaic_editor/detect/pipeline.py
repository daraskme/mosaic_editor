"""検出パイプライン — バックエンドの組み合わせと結果統合.

バックエンド:
- "sam3"        : SAM3 単体 (テキスト → マスク直接)。高速・推奨。
- "la_sam3"     : LocateAnything-3B (bbox) + SAM3 Tracker (マスク化)。
                  「挿入されたアナル」のような条件付き・文章的な概念に強い。
- "ensemble"    : 両方を実行して統合 (取りこぼし最小、最も遅い)。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.categories import Category
from ..core.masking import dilate_mask
from .base import Detection, ProgressCB, dedup_detections

BACKENDS: Dict[str, str] = {
    "sam3": "SAM3 のみ (高速・推奨)",
    "la_sam3": "LocateAnything-3B + SAM3 (条件付き概念に強い)",
    "ensemble": "両方を併用 (取りこぼし最小・低速)",
}


class DetectionPipeline:
    """モデルを遅延ロードしつつ各バックエンドを束ねる."""

    def __init__(self):
        self._sam3 = None
        self._refiner = None
        self._locator = None
        self._video_tracker = None

    # ---- 遅延ロード ----

    @property
    def sam3(self):
        if self._sam3 is None:
            from .sam3_concept import Sam3ConceptSegmenter
            self._sam3 = Sam3ConceptSegmenter()
        return self._sam3

    @property
    def refiner(self):
        if self._refiner is None:
            from .sam3_refine import Sam3BoxRefiner
            self._refiner = Sam3BoxRefiner()
        return self._refiner

    @property
    def locator(self):
        if self._locator is None:
            from .locate_anything import LocateAnythingDetector
            self._locator = LocateAnythingDetector()
        return self._locator

    @property
    def video_tracker(self):
        if self._video_tracker is None:
            from .sam3_video import Sam3VideoTracker
            self._video_tracker = Sam3VideoTracker()
        return self._video_tracker

    # ---- 画像検出 ----

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        backend: str = "sam3",
        threshold: float = 0.4,
        generation_mode: str = "hybrid",
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        detections: List[Detection] = []

        if backend in ("sam3", "ensemble"):
            detections.extend(self.sam3.detect(
                image, categories, threshold=threshold, progress_cb=progress_cb,
            ))

        if backend in ("la_sam3", "ensemble"):
            boxes = self.locator.detect(
                image, categories,
                generation_mode=generation_mode, progress_cb=progress_cb,
            )
            for i, det in enumerate(boxes):
                if progress_cb:
                    progress_cb(f"SAM3 で輪郭マスク化 [{i + 1}/{len(boxes)}]...")
                try:
                    det.mask = self.refiner.segment_box(image, det.bbox)
                except Exception as e:
                    print(f"[sam3_refine] failed for {det.bbox}: {e}")
                    det.mask = None
            detections.extend(boxes)

        return dedup_detections(detections)

    @staticmethod
    def combine_masks(
        detections: List[Detection],
        image_size: Tuple[int, int],
        margin_px: int = 0,
    ) -> np.ndarray:
        """検出結果を 1 枚の uint8 マスクに統合 (margin_px で外側に拡張)."""
        w, h = image_size
        out = np.zeros((h, w), dtype=np.uint8)
        for d in detections:
            if d.mask is not None and d.mask.shape == (h, w):
                out[d.mask > 127] = 255
            else:
                x1, y1, x2, y2 = d.bbox
                out[max(0, y1):y2, max(0, x1):x2] = 255
        if margin_px > 0:
            out = dilate_mask(out, margin_px)
        return out
