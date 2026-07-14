import json
import unittest
from unittest.mock import patch

import requests

from afac_pipeline.common.vlm_client import FinixDocClient, FinixDocTemporaryError


class FinixDocBusyResponseTest(unittest.TestCase):
    def test_http_200_busy_html_is_a_temporary_error(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response.headers["content-type"] = "text/html; charset=utf-8"
        response._content = "<title>服务器繁忙</title>顾客太多".encode()

        client = FinixDocClient(
            "https://example.invalid/api/finix_doc/call_with_file",
            user_id="finixA1001",
            api_key="test-key",
            max_retries=1,
        )
        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "服务器繁忙"):
                client.recognize(__file__, "")

    def test_temporary_error_type_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(FinixDocTemporaryError, RuntimeError))

    def test_retry_rotates_user_id_and_uses_logarithmic_delay(self) -> None:
        busy = requests.Response()
        busy.status_code = 200
        busy.headers["content-type"] = "text/html; charset=utf-8"
        busy._content = "<title>服务器繁忙</title>顾客太多".encode()

        success = requests.Response()
        success.status_code = 200
        success.headers["content-type"] = "application/json"
        success._content = json.dumps(
            {
                "success": True,
                "result": {
                    "result": json.dumps(
                        {"choices": [{"message": {"content": "# 成功"}}]},
                        ensure_ascii=False,
                    )
                },
            },
            ensure_ascii=False,
        ).encode()

        client = FinixDocClient(
            "https://example.invalid/api/finix_doc/call_with_file",
            user_id="finixB2002",
            user_ids=["finixB2002", "finixC3003", "finixA1001"],
            api_key="test-key",
            max_retries=2,
        )
        with (
            patch("requests.post", side_effect=[busy, busy, success]) as mocked_post,
            patch("afac_pipeline.common.vlm_client.time.sleep") as mocked_sleep,
        ):
            self.assertEqual(client.recognize(__file__, ""), "# 成功")

        used_users = [
            call.kwargs["data"]["userId"] for call in mocked_post.call_args_list
        ]
        self.assertEqual(
            used_users,
            ["finixB2002", "finixC3003", "finixA1001"],
        )
        # 第一次重试等待 1*log2(1)=0，因此不调用 sleep；
        # 第二次重试等待 2*log2(2)=2 秒。
        mocked_sleep.assert_called_once_with(2.0)


if __name__ == "__main__":
    unittest.main()
