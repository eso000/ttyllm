"""Tests for the chat buffer: message handling and imported data."""

import unittest
from typing import Any

from ai_chat.buffer import Buffer

DEFAULT_MODEL = "llama3.2"


class TestSetData(unittest.TestCase):
    """Tests for set data."""

    def _import(self, chat: dict[str, Any]) -> Buffer:
        """Test helper: import."""
        buffer = Buffer()
        buffer.set_data({"chat": chat})
        return buffer

    def test_valid_chat_is_loaded(self) -> None:
        """Valid chat is loaded."""
        buffer = self._import(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        self.assertEqual(buffer.chat_data["model"], "m")
        self.assertEqual(len(buffer.chat_data["messages"]), 3)

    def test_tool_calls_message_is_loaded(self) -> None:
        # regression: an assistant message carrying both content and
        # tool_calls used to be rejected, so exported chats would not import
        """Tool calls message is loaded."""
        buffer = self._import(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {"name": "x", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "id": "1", "content": "done"},
                ],
            }
        )
        self.assertEqual(buffer.chat_data["model"], "m")
        self.assertEqual(len(buffer.chat_data["messages"]), 3)

    def test_tool_calls_only_message_is_loaded(self) -> None:
        """Tool calls only message is loaded."""
        buffer = self._import(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {"name": "x", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            }
        )
        self.assertEqual(len(buffer.chat_data["messages"]), 1)

    def test_message_without_content_is_rejected(self) -> None:
        """Message without content is rejected."""
        buffer = self._import({"model": "m", "messages": [{"role": "user"}]})
        self.assertEqual(buffer.chat_data["model"], DEFAULT_MODEL)

    def test_bad_content_type_is_rejected(self) -> None:
        """Bad content type is rejected."""
        buffer = self._import(
            {"model": "m", "messages": [{"role": "user", "content": 5}]}
        )
        self.assertEqual(buffer.chat_data["model"], DEFAULT_MODEL)

    def test_missing_role_is_rejected(self) -> None:
        """Missing role is rejected."""
        buffer = self._import({"model": "m", "messages": [{"content": "hi"}]})
        self.assertEqual(buffer.chat_data["model"], DEFAULT_MODEL)

    def test_non_chat_payload_keeps_background_data(self) -> None:
        # no "chat" key: the payload itself is treated as the chat
        """Non chat payload keeps background data."""
        buffer = self._import(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(buffer.chat_data["model"], "m")


if __name__ == "__main__":
    unittest.main()
