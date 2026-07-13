import unittest

from afac_pipeline.long_models import SemanticPart, SemanticSegment
from afac_pipeline.long_packing import build_recognition_packs
from afac_pipeline.models import Box


def segment(index: int, start: int, end: int) -> SemanticSegment:
    part = SemanticPart(
        id=f"p{index}",
        segment_id=f"s{index}",
        role="h3_body",
        source_box=Box(0, start, 600, end),
        part_index=0,
        part_count=1,
        h1_id="body_h1",
        h2_id="h2_0000",
        h3_id=f"h3_{index:04d}",
        expected_heading_levels=(3,),
        file_name=f"p{index}.png",
    )
    return SemanticSegment(
        id=f"s{index}",
        role="h3_body",
        start_y=start,
        end_y=end,
        h1_id="body_h1",
        h2_id="h2_0000",
        h3_id=f"h3_{index:04d}",
        expected_heading_levels=(3,),
        parts=[part],
    )


class LongPackingTest(unittest.TestCase):
    def test_adjacent_small_sections_are_packed(self) -> None:
        segments = [segment(0, 0, 700), segment(1, 700, 1600), segment(2, 1600, 2600)]
        packs = build_recognition_packs(segments, image_width=600, max_height=3900)
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0].segment_ids, ("s0", "s1", "s2"))
        self.assertEqual(packs[0].source_box, Box(0, 0, 600, 2600))

    def test_pack_never_exceeds_limit(self) -> None:
        segments = [segment(0, 0, 2500), segment(1, 2500, 5000)]
        packs = build_recognition_packs(segments, image_width=600, max_height=3900)
        self.assertEqual(len(packs), 2)
        self.assertTrue(all(pack.source_box.height <= 3900 for pack in packs))


if __name__ == "__main__":
    unittest.main()
