from pathlib import Path
import json
import unittest
from unittest.mock import call, patch

import requests

from afac_pipeline.common.vlm_client import (
    FinixDocClient,
    FinixDocTemporaryError,
    parse_official_response,
)


def _json_response(payload: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.headers["content-type"] = "application/json"
    response._content = json.dumps(payload, ensure_ascii=False).encode()
    return response


def _successful_payload(content: str) -> dict:
    return {
        "success": True,
        "result": {
            "result": json.dumps(
                {"choices": [{"message": {"content": content}}]},
                ensure_ascii=False,
            )
        },
    }


class VlmRetryPolicyTest(unittest.TestCase):
    def test_explicit_recognition_failure_and_overload_are_temporary(self) -> None:
        for message in (
            "VL 识别失败，请稍后重试",
            "当前请求过载",
        ):
            with self.subTest(message=message):
                with self.assertRaises(FinixDocTemporaryError):
                    parse_official_response(
                        {"success": False, "message": message}
                    )

        with self.assertRaises(FinixDocTemporaryError):
            parse_official_response(_successful_payload("请求过载，请稍后重试"))

    def test_normal_empty_content_returns_blank_after_three_backoffs(self) -> None:
        empty_response = _json_response(_successful_payload(""))
        client = FinixDocClient(
            "https://example.invalid/api/finix_doc/call_with_file",
            user_id="finixA1001",
            user_ids=["finixA1001", "finixB2002"],
            api_key="test-key",
            max_retries=15,
        )
        with (
            patch(
                "requests.post",
                side_effect=[empty_response] * 4,
            ) as mocked_post,
            patch(
                "afac_pipeline.common.vlm_client.time.sleep"
            ) as mocked_sleep,
        ):
            markdown = client.recognize(
                Path(__file__),
                "",
                request_label="原图 table.jpg / 切块 tile.png",
                empty_retry_limit=3,
                return_empty_after_limit=True,
            )

        self.assertEqual(markdown, "")
        self.assertEqual(mocked_post.call_count, 4)
        self.assertEqual(
            mocked_sleep.call_args_list,
            [call(64.0), call(81.0), call(100.0)],
        )


if __name__ == "__main__":
    unittest.main()
