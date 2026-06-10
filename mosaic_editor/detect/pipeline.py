"""検出パイプライン — バックエンドの組み合わせと結果統合.

バックエンド:
- "la_sam2"     : LocateAnything-3B (bbox) + SAM2.1 (輪郭マスク化)。推奨・既定。
                  非gated なので HF の利用許諾なしで動く。
- "sam3"        : SAM3 単体 (テキスト → マスク直接)。gated (要 HF 同意)。
- "la_sam3"     : LocateAnything-3B (bbox) + SAM3 Tracker (マスク化)。gated。
- "ensemble"    : la_sam2 + sam3 を併用して統合 (取りこぼし最小、最も遅い)。

動画トラッキング:
- "la_sam2" 系 : LocateAnything でチャンク先頭を検出 → SAM2 Video で伝播
- "sam3" 系    : SAM3 Video がテキストプロンプトで検出 + 追跡
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.categories import Category
from ..core.masking import dilate_mask
from .base import Detection, ProgressCB, dedup_detections

BACKENDS: Dict[str, str] = {
    "la_sam2": "LocateAnything + SAM2 (推奨・HF同意不要)",
    "sam3": "SAM3 のみ (テキスト→マスク直接・要HF同意)",
    "la_sam3": "LocateAnything + SAM3 (要HF同意)",
    "ensemble": "LocateAnything+SAM2 と SAM3 を併用 (低速・取りこぼし最小)",
}

DEFAULT_BACKEND = "la_sam2"


class DetectionPipeline:
    """モデルを遅延ロードしつつ各バックエンドを束ねる."""

    def __init__(self):
        self._sam3 = None
        self._sam3_refiner = None
        self._sam2_refiner = None
        self._locator = None
        self._sam3_video = None
        self._la_sam2_video = None

    # ---- 遅延ロード ----

    @property
    def sam3(self):
        if self._sam3 is None:
            from .sam3_concept import Sam3ConceptSegmenter
            self._sam3 = Sam3ConceptSegmenter()
        return self._sam3

    @property
    def sam3_refiner(self):
        if self._sam3_refiner is None:
            from .sam3_refine import Sam3BoxRefiner
            self._sam3_refiner = Sam3BoxRefiner()
        return self._sam3_refiner

    @property
    def sam2_refiner(self):
        if self._sam2_refiner is None:
            from .sam2_refine import Sam2BoxRefiner
            self._sam2_refiner = Sam2BoxRefiner()
        return self._sam2_refiner

    @property
    def locator(self):
        if self._locator is None:
            from .locate_anything import LocateAnythingDetector
            self._locator = LocateAnythingDetector()
        return self._locator

    @property
    def sam3_video(self):
        if self._sam3_video is None:
            from .sam3_video import Sam3VideoTracker
            self._sam3_video = Sam3VideoTracker()
        return self._sam3_video

    @property
    def la_sam2_video(self):
        if self._la_sam2_video is None:
            from .sam2_video import LaSam2VideoTracker
            self._la_sam2_video = LaSam2VideoTracker(self.locator)
        return self._la_sam2_video

    # ---- 画像検出 ----

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        backend: str = DEFAULT_BACKEND,
        threshold: float = 0.4,
        generation_mode: str = "hybrid",
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        detections: List[Detection] = []

        if backend in ("la_sam2", "la_sam3", "ensemble"):
            refiner = (self.sam3_refiner if backend == "la_sam3"
                       else self.sam2_refiner)
            boxes = self.locator.detect(
                image, categories,
                generation_mode=generation_mode, progress_cb=progress_cb,
            )
            for i, det in enumerate(boxes):
                if progress_cb:
                    progress_cb(f"輪郭マスク化 [{i + 1}/{len(boxes)}]...")
                try:
                    det.mask = refiner.segment_box(image, det.bbox)
                except Exception as e:
                    print(f"[refine] failed for {det.bbox}: {e}")
                    det.mask = None
            detections.extend(boxes)

        if backend in ("sam3", "ensemble"):
            try:
                detections.extend(self.sam3.detect(
                    image, categories, threshold=threshold,
                    progress_cb=progress_cb,
                ))
            except Exception:
                if backend == "sam3":
                    raise
                # ensemble では SAM3 が使えなくても LA+SAM2 の結果を返す
                print("[ensemble] SAM3 が利用できないためスキップしました")

        return dedup_detections(detections)

    # ---- 動画トラッキング ----

    def track_video(
        self,
        video_path: str,
        categories: List[Category],
        backend: str = DEFAULT_BACKEND,
        generation_mode: str = "hybrid",
        progress_cb: ProgressCB = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[int, np.ndarray]:
        """バックエンドに応じた動画トラッキングを実行する."""
        if backend in ("sam3", "la_sam3"):
            return self.sam3_video.track_video(
                video_path, categories,
                progress_cb=progress_cb, cancel_check=cancel_check)
        return self.la_sam2_video.track_video(
            video_path, categories, generation_mode=generation_mode,
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
