"""均一な色のブロックが格子状に並ぶ領域を検出する (CPU のみ)."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image

from ..core.categories import MOSAIC_KEY, Category
from .base import Detection, ProgressCB


class MosaicDetector:
    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        threshold: float = 0.3,
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        category = next((c for c in categories if c.key == MOSAIC_KEY), None)
        if category is None:
            return []
        if progress_cb:
            progress_cb("既存のモザイク領域を検出中...")

        rgb = cv2.medianBlur(np.asarray(image.convert("RGB")), 3).astype(np.int16)
        h, w = rgb.shape[:2]
        if min(h, w) < 12:
            return []
        flat = np.ones((h, w), dtype=np.uint8)
        dx = np.max(np.abs(np.diff(rgb, axis=1)), axis=2) > 10
        dy = np.max(np.abs(np.diff(rgb, axis=0)), axis=2) > 10
        flat[:, :-1][dx] = 0
        flat[:, 1:][dx] = 0
        flat[:-1][dy] = 0
        flat[1:][dy] = 0
        _, _, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=4)
        blocks = stats[1:]
        bw, bh, area = blocks[:, 2], blocks[:, 3], blocks[:, 4]
        keep = (
            (bw >= 2) & (bh >= 2) & (bw <= 128) & (bh <= 128)
            & (np.minimum(bw, bh) >= 0.65 * np.maximum(bw, bh))
            & (area >= 0.8 * bw * bh)
        )
        blocks = blocks[keep]
        if len(blocks) < 6:
            return []

        sizes = np.sqrt((blocks[:, 2] + 2) * (blocks[:, 3] + 2))
        confidence = np.zeros((h, w), dtype=np.float32)
        for size in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
            group = blocks[(sizes >= size * 0.75) & (sizes <= size * 1.5)]
            if len(group) < 6:
                continue
            candidates = np.zeros((h, w), dtype=np.uint8)
            for x, y, bw, bh, _ in group:
                candidates[max(0, y - 1):min(h, y + bh + 1),
                           max(0, x - 1):min(w, x + bw + 1)] = 255
            gap = max(3, size // 2) | 1
            joined = cv2.morphologyEx(
                candidates, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
            count, labels, regions, _ = cv2.connectedComponentsWithStats(joined)
            centers = labels[group[:, 1] + group[:, 3] // 2,
                             group[:, 0] + group[:, 2] // 2]
            for region in range(1, count):
                members = group[centers == region]
                if len(members) < 6:
                    continue
                x, y, rw, rh, region_area = regions[region]
                mw, mh = np.median(members[:, 2:4] + 2, axis=0)
                if rw < 2.5 * mw or rh < 2.5 * mh:
                    continue
                coverage = float(region_area) / (rw * rh)
                score = min(0.99, len(members) / 12) * coverage
                if coverage < 0.5 or score < threshold:
                    continue
                local = labels[y:y + rh, x:x + rw] == region
                scores = confidence[y:y + rh, x:x + rw]
                scores[local] = np.maximum(scores[local], score)

        count, labels, regions, _ = cv2.connectedComponentsWithStats(
            (confidence > 0).astype(np.uint8))
        detections: List[Detection] = []
        for region in range(1, count):
            x, y, rw, rh, _ = regions[region]
            mask = (labels == region).astype(np.uint8) * 255
            detections.append(Detection(
                label=category.label,
                category_key=category.key,
                bbox=(int(x), int(y), int(x + rw), int(y + rh)),
                score=float(confidence[mask > 0].max()),
                mask=mask,
            ))
        return detections
