"""検出パイプライン — AnimeCensor (検出) + SAM2.1 (輪郭化・動画追跡).

- 画像: deepghs/anime_censor_detection (YOLOv8) で bbox 検出
        → SAM2.1 で輪郭マスク化
- 動画: チャンク先頭フレームを AnimeCensor で検出
        → SAM2.1 Video が全フレームに伝播 (追跡)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.categories import Category
from ..core.masking import dilate_mask
from .base import Detection, ProgressCB, dedup_detections


class DetectionPipeline:
    """モデルを遅延ロードしつつ検出・輪郭化・動画追跡を束ねる."""

    def __init__(self):
        self._anime = None
        self._refiner = None
        self._video_tracker = None

    # ---- 遅延ロード ----

    @property
    def anime(self):
        if self._anime is None:
            from .anime_censor import AnimeCensorDetector
            self._anime = AnimeCensorDetector()
        return self._anime

    @property
    def refiner(self):
        if self._refiner is None:
            from .sam2_refine import Sam2BoxRefiner
            self._refiner = Sam2BoxRefiner()
        return self._refiner

    # ---- 画像検出 ----

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        threshold: float = 0.3,
        use_refiner: bool = True,
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        boxes = self.anime.detect(
            image, categories, threshold=threshold, progress_cb=progress_cb)

        if use_refiner:
            for i, det in enumerate(boxes):
                if progress_cb:
                    progress_cb(f"SAM2 で輪郭マスク化 [{i + 1}/{len(boxes)}]...")
                try:
                    det.mask = self.refiner.segment_box(image, det.bbox)
                except Exception as e:
                    print(f"[refine] failed for {det.bbox}: {e}")
                    det.mask = None

        return dedup_detections(boxes)

    # ---- 動画トラッキング ----

    def track_video(
        self,
        video_path: str,
        categories: List[Category],
        threshold: float = 0.3,
        progress_cb: ProgressCB = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[int, np.ndarray]:
        if self._video_tracker is None:
            from .sam2_video import Sam2VideoTracker
            self._video_tracker = Sam2VideoTracker(lambda img, cats: [])
        self._video_tracker.detect_fn = lambda img, cats: self.anime.detect(
            img, cats, threshold=threshold)
        return self._video_tracker.track_video(
            video_path, categories,
            progress_cb=progress_cb, cancel_check=cancel_check)

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
