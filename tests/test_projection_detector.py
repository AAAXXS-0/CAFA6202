import unittest

from PIL import Image, ImageDraw

from afac_pipeline.config import TableConfig
from afac_pipeline.detectors import ProjectionTableDetector


class ProjectionDetectorTest(unittest.TestCase):
    def test_two_grid_tables_are_separated(self) -> None:
        image = Image.new("RGB", (600, 600), "white")
        draw = ImageDraw.Draw(image)
        for top in (40, 340):
            for y in range(top, top + 151, 50):
                draw.line((30, y, 570, y), fill="black", width=2)
            for x in range(30, 571, 90):
                draw.line((x, top, x, top + 150), fill="black", width=2)

        config = TableConfig(
            detector="projection",
            projection_min_line_ratio=0.5,
            projection_max_line_gap_ratio=0.1,
        )
        boxes = ProjectionTableDetector(config).detect(image)
        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0].box.y2, boxes[1].box.y1)


if __name__ == "__main__":
    unittest.main()
