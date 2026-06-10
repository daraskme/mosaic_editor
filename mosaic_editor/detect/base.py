"""検出バックエンドの共通インターフェース."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

ProgressCB = Optional[Callable[[str], None]]


@dataclass
class Detection:
    """1件の検出結果.

    bbox: (x1, y1, x2, y2) ピクセル座標
    mask: np.uint8 (H, W) 0/255 の輪郭マスク。box のみの場合 None。
    """
    label: str          # UI 表示用ラベル (カテゴリ label)
    category_key: str   # カテゴリの内部キー
    bbox: Tuple[int, int, int, int]
    score: float
    mask: Optional[np.ndarray] = None


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    import torch
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dedup_detections(detections: List[Detection], iou_thresh: float = 0.65) -> List[Detection]:
    """同一カテゴリ内で bbox IoU が高い重複を除去.

    輪郭マスク付きの検出をボックスのみの検出より優先し、
    同条件ではスコアの高い方を残す。
    """
    result: List[Detection] = []
    for d in sorted(detections, key=lambda x: (x.mask is None, -x.score)):
        dup = False
        for kept in result:
            if kept.category_key == d.category_key and _iou(kept.bbox, d.bbox) > iou_thresh:
                dup = True
                break
        if not dup:
            result.append(d)
    return result


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
