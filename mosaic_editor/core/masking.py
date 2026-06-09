"""マスク・モザイク処理のコアロジック (モデル非依存)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def auto_block_size(img_w: int, img_h: int) -> int:
    """審査規定準拠の自動ブロックサイズ.

    長辺が400px以上なら max(4, 長辺 // 100)、未満なら最小4px。
    """
    long_side = max(img_w, img_h)
    if long_side >= 400:
        return max(4, long_side // 100)
    return 4


def apply_mosaic(image_rgb: np.ndarray, mask: np.ndarray, block: int) -> np.ndarray:
    """mask>0 の領域にブロックモザイクを適用した画像を返す.

    モザイクのブロック格子は画像全体に固定 (マスク形状に依存しない) ので、
    隣接フレーム・隣接ストロークでも格子がぶれない。
    """
    h, w = image_rgb.shape[:2]
    block = max(2, block)
    small = cv2.resize(
        image_rgb,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = image_rgb.copy()
    mask_bool = mask > 0
    out[mask_bool] = mosaic_full[mask_bool]
    return out


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 alpha: float = 0.3) -> np.ndarray:
    """マスク領域に半透明カラーを重ねた表示用画像を返す."""
    mask_bool = mask > 0
    if not mask_bool.any():
        return image_rgb
    out = image_rgb.copy()
    colored = out.copy()
    colored[mask_bool] = color
    blended = cv2.addWeighted(out, 1.0 - alpha, colored, alpha, 0)
    out[mask_bool] = blended[mask_bool]
    return out


def dilate_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """検出マスクを margin_px だけ外側に膨張させる (取りこぼし防止)."""
    if margin_px <= 0:
        return mask
    k = 2 * margin_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def merge_masks(base: Optional[np.ndarray], add: np.ndarray) -> np.ndarray:
    if base is None:
        return add.copy()
    return np.maximum(base, add)
