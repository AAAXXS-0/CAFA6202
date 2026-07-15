import unittest

from PIL import Image, ImageDraw

from afac_pipeline.table.config import TableConfig
from afac_pipeline.table.detectors import InkTableDetector, create_detector
from afac_pipeline.table.ink_region import detect_ink_regions


class InkRegionTest(unittest.TestCase):
    def test_auto_detector_uses_model_free_ink_regions(self) -> None:
        detector = create_detector(TableConfig(detector="auto"))
        self.assertIsInstance(detector, InkTableDetector)

    def test_sparse_trapezoid_text_is_joined_into_complete_region(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        draw = ImageDraw.Draw(image)
        for row, y in enumerate(range(130, 591, 55)):
            left = 150 - row * 5
            right = 820 + row * 7
            for x in range(left, right, 95):
                draw.rectangle((x, y, min(x + 55, right), y + 15), fill="black")

        result = detect_ink_regions(image, coarse_max_side=256)
        self.assertTrue(result.regions)
        primary = result.regions[0].box
        self.assertLessEqual(primary.x1, 170)
        self.assertGreaterEqual(primary.x2, 850)
        self.assertLessEqual(primary.y1, 150)
        self.assertGreaterEqual(primary.y2, 590)

    def test_fixed_coarse_scale_does_not_depend_on_maximum_side(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        result = detect_ink_regions(
            image, coarse_max_side=64, coarse_scale=0.25
        )
        self.assertEqual(result.coarse_size, (250, 175))


if __name__ == "__main__":
    unittest.main()
