"""検出パイプライン — モザイク格子 / AnimeCensor + SAM2.1.

- 画像: deepghs/anime_censor_detection (YOLOv8) で bbox 検出
        → SAM2.1 で輪郭マスク化
- 動画: チャンク先頭フレームを AnimeCensor で検出
        → SAM2.1 Video が全フレームに伝播 (追跡)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.categories import MOSAIC_KEY, Category, split_categories
from ..core.masking import dilate_mask
from .base import Detection, ProgressCB, dedup_detections


class DetectionPipeline:
    """モデルを遅延ロードしつつ検出・輪郭化・動画追跡を束ねる."""

    def __init__(self):
        self._anime = None
        self._mosaic = None
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
    def mosaic(self):
        if self._mosaic is None:
            from .mosaic import MosaicDetector
            self._mosaic = MosaicDetector()
        return self._mosaic

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
        anime_categories, mosaic_categories = split_categories(categories)
        boxes: List[Detection] = []
        if anime_categories:
            boxes = self.anime.detect(
                image, anime_categories, threshold=threshold,
                progress_cb=progress_cb)

        if use_refiner and boxes:
            for i, det in enumerate(boxes):
                if progress_cb:
                    progress_cb(f"SAM2 で輪郭マスク化 [{i + 1}/{len(boxes)}]...")
                try:
                    det.mask = self.refiner.segment_box(image, det.bbox)
                except Exception as e:
                    print(f"[refine] failed for {det.bbox}: {e}")
                    det.mask = None

        mosaics: List[Detection] = []
        if mosaic_categories:
            mosaics = self.mosaic.detect(
                image, mosaic_categories, threshold=threshold,
                progress_cb=progress_cb)
        return dedup_detections(boxes + mosaics)

    # ---- 動画トラッキング ----

    def track_video(
        self,
        video_path: str,
        categories: List[Category],
        threshold: float = 0.3,
        progress_cb: ProgressCB = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[int, np.ndarray]:
        anime_categories, mosaic_categories = split_categories(categories)
        masks: Dict[int, np.ndarray] = {}
        if anime_categories:
            if self._video_tracker is None:
                from .sam2_video import Sam2VideoTracker
                self._video_tracker = Sam2VideoTracker()
            masks = self._video_tracker.track_video(
                video_path, anime_categories,
                detect_fn=lambda img, cats: self.anime.detect(
                    img, cats, threshold=threshold),
                progress_cb=progress_cb, cancel_check=cancel_check)
        if mosaic_categories:
            mosaic_masks = self._detect_video_mosaics(
                video_path, mosaic_categories, threshold, progress_cb,
                cancel_check)
            for frame_idx, mask in mosaic_masks.items():
                existing = masks.get(frame_idx)
                masks[frame_idx] = np.maximum(existing, mask) \
                    if existing is not None else mask
        return masks

    def _detect_video_mosaics(
        self,
        video_path: str,
        categories: List[Category],
        threshold: float,
        progress_cb: ProgressCB,
        cancel_check: Optional[Callable[[], bool]],
    ) -> Dict[int, np.ndarray]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        masks: Dict[int, np.ndarray] = {}
        try:
            frame_idx = 0
            while True:
                if cancel_check and cancel_check():
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                detections = self.mosaic.detect(
                    image, categories, threshold=threshold)
                if detections:
                    masks[frame_idx] = self.combine_masks(
                        detections, image.size)
                if progress_cb and frame_idx % 20 == 0:
                    progress_cb(
                        f"既存モザイクを検出中... フレーム {frame_idx + 1}/{total}")
                frame_idx += 1
        finally:
            cap.release()
        return masks

    @staticmethod
    def required_packages(categories: List[Category]) -> Tuple[str, ...]:
        """選択カテゴリの検出に必要な追加パッケージ (モザイクのみなら不要)."""
        if any(c.key != MOSAIC_KEY for c in categories):
            return "torch", "transformers", "imgutils"
        return ()

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
