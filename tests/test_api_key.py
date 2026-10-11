import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

from ai_chat.api_key import ApiKey, ApiKeyType


@contextmanager
def _without_env(name: str):
    """temporarily remove an environment variable so it reads as unset"""
    saved = os.environ.pop(name, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[name] = saved


class TestRawKey(unittest.TestCase):
    def test_raw_key_is_returned(self) -> None:
        self.assertEqual(ApiKey("sk-123", ApiKeyType.RAW).get_key(), "sk-123")

    def test_raw_key_is_stripped(self) -> None:
        self.assertEqual(ApiKey("  sk-123\n", ApiKeyType.RAW).get_key(), "sk-123")

    def test_empty_raw_key_is_none(self) -> None:
        self.assertIsNone(ApiKey("   ", ApiKeyType.RAW).get_key())

    def test_key_with_embedded_newline_is_rejected(self) -> None:
        self.assertIsNone(ApiKey("sk-1\r\n23", ApiKeyType.RAW).get_key())

    def test_repr_hides_the_key(self) -> None:
        self.assertNotIn("sk-123", repr(ApiKey("sk-123", ApiKeyType.RAW)))


class TestEnvKey(unittest.TestCase):
    def test_env_var_is_returned(self) -> None:
        with mock.patch.dict(os.environ, {"AI_CHAT_TEST_KEY": "sk-abc"}):
            self.assertEqual(
                ApiKey("AI_CHAT_TEST_KEY", ApiKeyType.ENV).get_key(), "sk-abc"
            )

    def test_env_var_is_stripped(self) -> None:
        with mock.patch.dict(os.environ, {"AI_CHAT_TEST_KEY": "sk-abc\n"}):
            self.assertEqual(
                ApiKey("AI_CHAT_TEST_KEY", ApiKeyType.ENV).get_key(), "sk-abc"
            )

    def test_unset_env_var_is_none(self) -> None:
        with _without_env("AI_CHAT_TEST_KEY"):
            self.assertIsNone(ApiKey("AI_CHAT_TEST_KEY", ApiKeyType.ENV).get_key())

    def test_empty_env_var_is_none(self) -> None:
        with mock.patch.dict(os.environ, {"AI_CHAT_TEST_KEY": "  "}):
            self.assertIsNone(ApiKey("AI_CHAT_TEST_KEY", ApiKeyType.ENV).get_key())


class TestFileKey(unittest.TestCase):
    def write_file(self, content: str) -> str:
        """write a key file in a temp dir and return its path"""
        handle = tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8"
        )
        handle.write(content)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_file_contents_are_returned(self) -> None:
        path = self.write_file("sk-file-1")
        self.assertEqual(ApiKey(path, ApiKeyType.FILE).get_key(), "sk-file-1")

    def test_trailing_newline_is_stripped(self) -> None:
        path = self.write_file("sk-file-1\n")
        self.assertEqual(ApiKey(path, ApiKeyType.FILE).get_key(), "sk-file-1")

    def test_missing_file_is_none(self) -> None:
        self.assertIsNone(ApiKey("/nonexistent/key/file", ApiKeyType.FILE).get_key())

    def test_directory_is_none(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, directory)
        self.assertIsNone(ApiKey(directory, ApiKeyType.FILE).get_key())

    def test_empty_file_is_none(self) -> None:
        path = self.write_file(" \n ")
        self.assertIsNone(ApiKey(path, ApiKeyType.FILE).get_key())

    def test_key_with_embedded_newline_is_none(self) -> None:
        path = self.write_file("sk-1\n23")
        self.assertIsNone(ApiKey(path, ApiKeyType.FILE).get_key())


if __name__ == "__main__":
    unittest.main()