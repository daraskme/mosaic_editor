"""既存モザイク検出とカテゴリ振り分けのテスト."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from mosaic_editor.core.categories import (DEFAULT_CATEGORIES, MOSAIC_KEY,
                                           split_categories)
from mosaic_editor.detect.mosaic import MosaicDetector
from mosaic_editor.detect.pipeline import DetectionPipeline

MOSAIC_CATEGORIES = [c for c in DEFAULT_CATEGORIES if c.key == MOSAIC_KEY]


def make_image(seed: int = 0) -> np.ndarray:
    """ブロックより細かい構造を持たない、コントラストの高い下地を作る."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(coarse).resize((256, 256), Image.Resampling.BICUBIC))


def pixelate(rgb: np.ndarray, box: tuple, block: int) -> np.ndarray:
    out = rgb.copy()
    x1, y1, x2, y2 = box
    for y in range(y1, y2, block):
        for x in range(x1, x2, block):
            cell = out[y:y + block, x:x + block]
            cell[:] = cell.mean(axis=(0, 1)).astype(np.uint8)
    return out


class MosaicDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = MosaicDetector()

    def detect(self, rgb: np.ndarray):
        return self.detector.detect(Image.fromarray(rgb), MOSAIC_CATEGORIES)

    def test_detects_block_mosaic_region(self) -> None:
        for block in (8, 12, 16):
            with self.subTest(block=block):
                rgb = pixelate(make_image(), (64, 48, 192, 176), block)
                detections = self.detect(rgb)
                self.assertTrue(detections)
                best = max(detections, key=lambda d: d.score)
                x1, y1, x2, y2 = best.bbox
                self.assertEqual(best.category_key, MOSAIC_KEY)
                self.assertGreater(best.score, 0.3)
                self.assertLess(abs(x1 - 64), block + 4)
                self.assertLess(abs(y1 - 48), block + 4)
                self.assertLess(abs(x2 - 192), block + 4)
                self.assertLess(abs(y2 - 176), block + 4)
                self.assertEqual(best.mask.shape, rgb.shape[:2])
                self.assertEqual(sorted(np.unique(best.mask)), [0, 255])

    def test_no_detection_on_smooth_image(self) -> None:
        self.assertEqual(self.detect(make_image(1)), [])

    def test_empty_when_mosaic_not_selected(self) -> None:
        rgb = pixelate(make_image(), (32, 32, 224, 224), 24)
        others = [c for c in DEFAULT_CATEGORIES if c.key != MOSAIC_KEY]
        self.assertEqual(self.detector.detect(Image.fromarray(rgb), others), [])

    def test_tiny_image_is_ignored(self) -> None:
        self.assertEqual(self.detect(np.zeros((8, 8, 3), np.uint8)), [])


class CategoryRoutingTest(unittest.TestCase):
    def test_split_categories(self) -> None:
        anime, mosaic = split_categories(DEFAULT_CATEGORIES)
        self.assertEqual([c.key for c in mosaic], [MOSAIC_KEY])
        self.assertNotIn(MOSAIC_KEY, [c.key for c in anime])

    def test_required_packages(self) -> None:
        self.assertEqual(
            DetectionPipeline.required_packages(MOSAIC_CATEGORIES), ())
        self.assertEqual(
            DetectionPipeline.required_packages(DEFAULT_CATEGORIES),
            ("torch", "transformers", "imgutils"))


if __name__ == "__main__":
    unittest.main()
