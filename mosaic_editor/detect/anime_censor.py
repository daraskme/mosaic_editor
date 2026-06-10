"""deepghs/anime_censor_detection (imgutils) バックエンド — イラスト/アニメ絵向け.

booru 系の大規模アノテーションで学習された YOLOv8 検出器。
アニメ絵特有のデフォルメ描写・結合部に強く、ONNX 実行で1枚数十ms と高速。

ラベル: nipple_f (女性乳首) / penis (男性器) / pussy (女性器)
"""
from __future__ import annotations

from typing import Dict, List

from PIL import Image

from ..core.categories import Category
from .base import Detection, ProgressCB

# モデルラベル → カテゴリ key
LABEL_TO_CATEGORY_KEY: Dict[str, str] = {
    "penis": "penis",
    "pussy": "vagina",
    "nipple_f": "nipples",
}

# このバックエンドが対応するカテゴリ key
SUPPORTED_CATEGORY_KEYS = set(LABEL_TO_CATEGORY_KEY.values())


class AnimeCensorDetector:
    """imgutils.detect.detect_censors のラッパー (遅延 import)."""

    def __init__(self, level: str = "s"):
        # level: 's' (標準・高精度) / 'n' (nano・高速)
        self.level = level

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        threshold: float = 0.3,
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        from imgutils.detect import detect_censors

        if image.mode != "RGB":
            image = image.convert("RGB")

        # 選択カテゴリのうちこのモデルで検出可能なもの
        wanted: Dict[str, Category] = {
            c.key: c for c in categories if c.key in SUPPORTED_CATEGORY_KEYS}
        if not wanted:
            return []

        if progress_cb:
            progress_cb("AnimeCensor (YOLOv8) 検出中...")

        results = detect_censors(image, level=self.level,
                                 conf_threshold=threshold)

        detections: List[Detection] = []
        for (x0, y0, x1, y1), label, score in results:
            key = LABEL_TO_CATEGORY_KEY.get(label)
            if key is None or key not in wanted:
                continue
            cat = wanted[key]
            detections.append(Detection(
                label=cat.label,
                category_key=cat.key,
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                score=float(score),
            ))
        return detections
