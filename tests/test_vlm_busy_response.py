from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Lock
import unittest
from unittest.mock import call, patch

import requests

from afac_pipeline.common.vlm_client import (
    FinixDocClient,
    FinixDocTemporaryError,
    MAX_RETRY_COUNT,
    retry_delay_seconds,
)


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
        with (
            patch("requests.post", return_value=response),
            patch("afac_pipeline.common.vlm_client.time.sleep") as mocked_sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "服务器繁忙"):
                client.recognize(__file__, "")
        mocked_sleep.assert_called_once_with(64.0)

    def test_temporary_error_type_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(FinixDocTemporaryError, RuntimeError))

    def test_retry_rotates_user_id_and_uses_quadratic_delay(self) -> None:
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
        # 平方退让从 8² 开始，随后是 9²。
        self.assertEqual(
            mocked_sleep.call_args_list,
            [call(64.0), call(81.0)],
        )

    def test_empty_model_content_rotates_user_and_retries(self) -> None:
        empty = requests.Response()
        empty.status_code = 200
        empty.headers["content-type"] = "application/json"
        empty._content = json.dumps(
            {
                "success": False,
                "message": "direct response content is empty",
            }
        ).encode()

        success = requests.Response()
        success.status_code = 200
        success.headers["content-type"] = "application/json"
        success._content = json.dumps(
            {
                "success": True,
                "result": {
                    "result": json.dumps(
                        {"choices": [{"message": {"content": "# 重试成功"}}]},
                        ensure_ascii=False,
                    )
                },
            },
            ensure_ascii=False,
        ).encode()

        client = FinixDocClient(
            "https://example.invalid/api/finix_doc/call_with_file",
            user_id="finixA1001",
            user_ids=["finixA1001", "finixB2002"],
            api_key="test-key",
            max_retries=1,
        )
        with (
            patch("requests.post", side_effect=[empty, success]) as mocked_post,
            patch("afac_pipeline.common.vlm_client.time.sleep") as mocked_sleep,
        ):
            self.assertEqual(client.recognize(__file__, ""), "# 重试成功")

        self.assertEqual(
            [item.kwargs["data"]["userId"] for item in mocked_post.call_args_list],
            ["finixA1001", "finixB2002"],
        )
        mocked_sleep.assert_called_once_with(64.0)

    def test_parallel_requests_rotate_their_initial_user_ids(self) -> None:
        users = ["finixA1001", "finixB2002", "finixC3003"]
        client = FinixDocClient(
            "https://example.invalid/api",
            user_id=users[0],
            user_ids=users,
            api_key="test-key",
            max_retries=0,
        )
        seen: list[str | None] = []
        lock = Lock()

        def recognize_once(*args, active_user_id=None, **kwargs):
            with lock:
                seen.append(active_user_id)
            return f"# {active_user_id}"

        with patch.object(client, "_recognize_once", side_effect=recognize_once):
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(
                    executor.map(
                        lambda _: client.recognize(__file__, ""),
                        range(6),
                    )
                )

        self.assertEqual(len(results), 6)
        self.assertEqual(
            Counter(seen),
            Counter({"finixA1001": 2, "finixB2002": 2, "finixC3003": 2}),
        )

    def test_quadratic_delay_sequence_starts_at_eight_squared(self) -> None:
        self.assertEqual(
            [retry_delay_seconds(index) for index in range(1, 5)],
            [64.0, 81.0, 100.0, 121.0],
        )

    def test_retry_limit_is_hard_capped_at_fifteen(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 到 15"):
            FinixDocClient(
                "https://example.invalid/api",
                api_key="test-key",
                user_id="finixA1001",
                max_retries=MAX_RETRY_COUNT + 1,
            )

        client = FinixDocClient(
            "https://example.invalid/api",
            api_key="test-key",
            user_id="finixA1001",
            max_retries=MAX_RETRY_COUNT,
        )
        with (
            patch.object(
                client,
                "_recognize_once",
                side_effect=FinixDocTemporaryError("服务器繁忙"),
            ) as mocked_once,
            patch("afac_pipeline.common.vlm_client.time.sleep") as mocked_sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "服务器繁忙"):
                client.recognize(__file__, "")

        # 首次请求加 15 次重试，共尝试 16 次；第 15 次等待 (8+14)²=484 秒。
        self.assertEqual(mocked_once.call_count, 16)
        self.assertEqual(mocked_sleep.call_count, 15)
        self.assertEqual(mocked_sleep.call_args_list[0], call(64.0))
        self.assertEqual(mocked_sleep.call_args_list[-1], call(484.0))


if __name__ == "__main__":
    unittest.main()
