import unittest

from PIL import Image, ImageDraw

from afac_pipeline.table.detectors import find_content_box


class ContentBoxTest(unittest.TestCase):
    def test_tiny_isolated_footer_is_ignored(self) -> None:
        image = Image.new("RGB", (600, 800), "white")
        draw = ImageDraw.Draw(image)
        # 主体表格占据页面上半部，页脚只有极少墨迹。
        for y in range(80, 381, 50):
            draw.line((60, y, 540, y), fill="black", width=2)
        for x in range(60, 541, 80):
            draw.line((x, 80, x, 380), fill="black", width=2)
        draw.rectangle((290, 760, 300, 762), fill="black")

        box = find_content_box(image)
        self.assertLess(box.y2, 500)
        self.assertGreater(box.width, 400)


if __name__ == "__main__":
    unittest.main()
