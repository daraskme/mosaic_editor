"""画像特徴を共有する輪郭化の接続と矩形フォールバックを検証."""
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from mosaic_editor.core.categories import DEFAULT_CATEGORIES
from mosaic_editor.detect.base import Detection
from mosaic_editor.detect.pipeline import DetectionPipeline


class RefinementPipelineTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 24))
        self.categories = [c for c in DEFAULT_CATEGORIES if c.key == "vagina"]
        self.boxes = [
            Detection("女性器", "vagina", (2, 3, 10, 12), 0.8),
            Detection("女性器", "vagina", (18, 4, 28, 20), 0.7),
        ]
        self.pipeline = DetectionPipeline()
        self.anime = Mock()
        self.anime.detect.return_value = self.boxes
        self.refiner = Mock()
        self.pipeline._anime = self.anime
        self.pipeline._refiner = self.refiner

    def test_refines_all_boxes_in_one_call_preserving_order(self):
        masks = [np.zeros((24, 32), np.uint8) for _ in self.boxes]
        masks[0][4:10, 3:9] = 255
        masks[1][5:18, 19:27] = 255
        self.refiner.segment_boxes.return_value = masks
        progress = Mock()

        detections = self.pipeline.detect(
            self.image, self.categories, progress_cb=progress)

        self.refiner.segment_boxes.assert_called_once_with(
            self.image, [d.bbox for d in self.boxes], progress_cb=progress)
        self.assertIs(detections[0].mask, masks[0])
        self.assertIs(detections[1].mask, masks[1])
        np.testing.assert_array_equal(
            self.pipeline.combine_masks(detections, self.image.size),
            np.maximum(*masks),
        )

    def test_load_failure_keeps_boxes_without_repeated_load_attempts(self):
        self.refiner.segment_boxes.side_effect = OSError("download interrupted")
        progress = Mock()

        with self.assertLogs("mosaic_editor.detect.pipeline", level="WARNING"):
            detections = self.pipeline.detect(
                self.image, self.categories, progress_cb=progress)

        self.refiner.segment_boxes.assert_called_once()
        self.assertEqual(detections, self.boxes)
        self.assertTrue(all(d.mask is None for d in detections))
        self.assertIn("download interrupted", progress.call_args.args[0])
        mask = self.pipeline.combine_masks(detections, self.image.size)
        self.assertEqual(np.count_nonzero(mask), 8 * 9 + 10 * 16)

    def test_one_failed_box_keeps_the_other_contour(self):
        contour = np.zeros((24, 32), np.uint8)
        contour[4:10, 3:9] = 255
        self.refiner.segment_boxes.return_value = [contour, None]

        detections = self.pipeline.detect(self.image, self.categories)

        mask = self.pipeline.combine_masks(detections, self.image.size)
        self.assertEqual(np.count_nonzero(mask), 6 * 6 + 10 * 16)
        self.assertEqual(mask[3, 2], 0)

    def test_no_boxes_or_disabled_refinement_does_not_load_sam(self):
        self.pipeline.detect(self.image, self.categories, use_refiner=False)
        self.refiner.segment_boxes.assert_not_called()
        self.anime.detect.return_value = []
        self.assertEqual(self.pipeline.detect(self.image, self.categories), [])
        self.refiner.segment_boxes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
