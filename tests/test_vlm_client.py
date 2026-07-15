import unittest

from afac_pipeline.common.vlm_client import (
    CHAT_PROTOCOL,
    FinixDocTemporaryError,
    OFFICIAL_PROTOCOL,
    parse_official_response,
    select_request_prompt,
)


class FinixDocClientTest(unittest.TestCase):
    def test_official_request_does_not_even_build_a_prompt(self) -> None:
        calls = 0

        def build_prompt() -> str:
            nonlocal calls
            calls += 1
            return "不应生成"

        self.assertEqual(select_request_prompt(OFFICIAL_PROTOCOL, build_prompt), "")
        self.assertEqual(calls, 0)
        self.assertEqual(select_request_prompt(CHAT_PROTOCOL, build_prompt), "不应生成")
        self.assertEqual(calls, 1)

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
