import unittest

from afac_pipeline.common.vlm_client import (
    FinixDocTemporaryError,
    parse_official_response,
)


class FinixDocClientTest(unittest.TestCase):
    def test_official_nested_response_and_fence_are_parsed(self) -> None:
        payload = {
            "success": True,
            "result": {
                "result": (
                    '{"choices":[{"message":{"content":'
                    '"```markdown\\n# 标题\\n\\n正文\\n```"}}]}'
                )
            },
        }

        self.assertEqual(parse_official_response(payload), "# 标题\n\n正文")

    def test_official_business_error_is_not_silenced(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "无权限"):
            parse_official_response({"success": False, "message": "无权限"})

    def test_official_empty_content_is_temporary(self) -> None:
        message = (
            "请求发生错误：FinixDoc parse_channel=new_openai_compatible "
            "direct response content is empty"
        )
        with self.assertRaisesRegex(FinixDocTemporaryError, "临时空响应"):
            parse_official_response({"success": False, "message": message})


if __name__ == "__main__":
    unittest.main()
