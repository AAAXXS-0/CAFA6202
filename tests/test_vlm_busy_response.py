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


if __name__ == "__main__":
    unittest.main()
