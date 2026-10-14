"""Tests for tool-call collection, approval and execution."""

import unittest
from typing import Any
from unittest import mock

from ai_chat import toolchain
from ai_chat.buffer import Buffer
from ai_chat.requester import Requester
from ai_chat.tool import Parameter, Tool
from test_requester import _http_response, _serve_once

def _call(
    name: str = "echo", args: Any = None, call_id: str = "call_1"
) -> dict[str, Any]:
    """Test helper: call."""
    return {"id": call_id, "function": {"name": name, "arguments": args or {}}}

class ToolTestCase(unittest.TestCase):
    """Shared test case for tool."""
    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        self.executed: list[str] = []

        def echo(text: str = "", **_: Any) -> str:
            """Test helper: echo."""
            self.executed.append(text)
            return f"echoed {text}"

        tool = Tool("echo", echo, "echo", [Parameter("text", "string", "text")])
        patcher = mock.patch.object(toolchain, "toolchain", [tool])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chat = Buffer()
        self.chat.in_request = True
        self.chat.add_msg("assistant")
        self.requester = Requester()

class TestCollectingToolCalls(ToolTestCase):
    """Tests for collecting tool calls."""
    def test_tool_is_not_run_before_approval(self) -> None:
        """Tool is not run before approval."""
        self.requester._handle_tool_call(self.chat, [_call(args={"text": "hi"})])
        self.assertEqual(self.executed, [])
        self.assertTrue(self.chat.awaiting_approval)
        self.assertEqual(self.chat.pending_calls[0]["args"], {"text": "hi"})

    def test_json_string_arguments_are_parsed(self) -> None:
        """Json string arguments are parsed."""
        self.requester._handle_tool_call(self.chat, [_call(args='{"text": "hi"}')])
        self.assertEqual(self.chat.pending_calls[0]["args"], {"text": "hi"})

    def test_broken_arguments_become_empty(self) -> None:
        """Broken arguments become empty."""
        self.requester._handle_tool_call(self.chat, [_call(args="{nope")])
        self.assertEqual(self.chat.pending_calls[0]["args"], {})

    def test_unknown_tool_is_rejected_without_asking(self) -> None:
        """Unknown tool is rejected without asking."""
        self.requester._handle_tool_call(self.chat, [_call(name="rm_rf")])
        self.assertFalse(self.chat.awaiting_approval)
        self.assertTrue(self.chat.pending_resolved())
        self.assertIn("does not exist", self.chat.pending_calls[0]["result"])

    def test_missing_call_id_is_tolerated(self) -> None:
        """Missing call id is tolerated."""
        call = {"function": {"name": "echo", "arguments": {}}}
        self.requester._handle_tool_call(self.chat, [call])
        self.assertIsNone(self.chat.pending_calls[0]["id"])

class TestDeciding(ToolTestCase):
    """Tests for deciding."""
    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        super().setUp()
        self.requester._handle_tool_call(
            self.chat,
            [
                _call(args={"text": "a"}, call_id="1"),
                _call(args={"text": "b"}, call_id="2"),
            ],
        )
        self.chat.in_request = False

    def test_calls_are_decided_in_order(self) -> None:
        """Calls are decided in order."""
        self.assertFalse(self.chat.decide(True))
        self.assertTrue(self.chat.awaiting_approval)
        self.assertTrue(self.chat.decide(False))
        self.assertEqual(
            [c["approved"] for c in self.chat.pending_calls], [True, False]
        )

    def test_resume_runs_only_approved_calls(self) -> None:
        """Resume runs only approved calls."""
        self.chat.decide(True)
        self.chat.decide(False)
        with mock.patch.object(self.requester, "get_reply", return_value=None) as cont:
            self.assertIsNone(self.requester.resume(self.chat))
        cont.assert_called_once_with(self.chat, wrapped=True)
        self.assertEqual(self.executed, ["a"])
        tool_msgs = [m for m in self.chat.chat_data["messages"] if m["role"] == "tool"]
        self.assertEqual(
            [(m["id"], m["content"]) for m in tool_msgs],
            [("1", "echoed a"), ("2", "user declined call")],
        )
        self.assertEqual(self.chat.pending_calls, [])

    def test_undecided_call_is_never_run(self) -> None:
        """Undecided call is never run."""
        self.chat.decide(True)
        with mock.patch.object(self.requester, "get_reply", return_value=None):
            self.requester.resume(self.chat)
        self.assertEqual(self.executed, ["a"])

    def test_failing_tool_reports_error_to_model(self) -> None:
        """Failing tool reports error to model."""
        toolchain.toolchain[0].function = mock.Mock(side_effect=RuntimeError("boom"))
        self.chat.decide(True)
        self.chat.decide(True)
        with mock.patch.object(self.requester, "get_reply", return_value=None):
            self.requester.resume(self.chat)
        contents = [
            m["content"] for m in self.chat.chat_data["messages"] if m["role"] == "tool"
        ]
        self.assertEqual(contents, ["tool call failed: boom"] * 2)

    def test_abort_declines_everything_and_reopens_input(self) -> None:
        """Abort declines everything and reopens input."""
        self.assertTrue(self.chat.abort_pending())
        messages = self.chat.chat_data["messages"]
        self.assertEqual([m["role"] for m in messages[-3:]], ["tool", "tool", "user"])
        self.assertEqual(messages[-3]["content"], "user declined call")
        self.assertEqual(self.executed, [])
        self.assertFalse(self.chat.abort_pending())

    def test_abort_is_refused_while_a_request_runs(self) -> None:
        """Abort is refused while a request runs."""
        self.chat.in_request = True
        self.assertFalse(self.chat.abort_pending())
        self.assertEqual(len(self.chat.pending_calls), 2)

class TestBufferWhilePending(ToolTestCase):
    """Tests for buffer while pending."""
    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        super().setUp()
        self.requester._handle_tool_call(self.chat, [_call()])
        self.chat.in_request = False

    def test_typing_is_blocked(self) -> None:
        """Typing is blocked."""
        before = self.chat.chat_data["messages"][-1]["content"]
        self.chat.add_open("x")
        self.chat.del_open()
        self.chat.change_msg("user")
        self.assertEqual(self.chat.chat_data["messages"][-1]["content"], before)
        self.assertEqual(self.chat.chat_data["messages"][-1]["role"], "assistant")
        idle, _ = self.chat.currently_typed()
        self.assertFalse(idle)

    def test_prompt_is_rendered_in_the_chat(self) -> None:
        """Prompt is rendered in the chat."""
        self.chat.pending_calls[0]["args"] = {"text": "ls\x1b[2J"}
        text = "\n".join(self.chat.get_text_box(80).lines)
        self.assertIn("approval required", text)
        self.assertIn("[y] approve", text)
        self.assertNotIn("\x1b[2J", text)  # control characters are escaped

    def test_clearing_the_chat_drops_pending_calls(self) -> None:
        """Clearing the chat drops pending calls."""
        self.chat.empty()
        self.assertEqual(self.chat.pending_calls, [])

class TestRequestEndsAtToolCall(ToolTestCase):
    """Tests for request ends at tool call."""
    def test_request_terminates_and_waits_for_approval(self) -> None:
        """Request terminates and waits for approval."""
        chat = Buffer()
        body = {
            "message": {"content": "", "tool_calls": [_call(args={"text": "hi"})]},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
        port, _ = _serve_once(_http_response(body))
        requester = Requester(port=port)
        chat.add_open("x")
        chat.change_msg("user")
        self.assertIsNone(requester.get_reply(chat))

        self.assertFalse(chat.in_request)
        self.assertTrue(chat.awaiting_approval)
        self.assertEqual(self.executed, [])
        self.assertEqual(chat.chat_data["messages"][-1]["role"], "assistant")
        self.assertIn("tool_calls", chat.chat_data["messages"][-1])

    def test_tool_chain_is_capped_per_buffer(self) -> None:
        """Tool chain is capped per buffer."""
        chat = Buffer()
        chat.tool_depth = Requester.MAX_TOOL_DEPTH
        requester = Requester()
        error = requester.get_reply(chat, wrapped=True)
        self.assertIn("exceeded", error)  # type: ignore[arg-type]
        self.assertEqual(chat.chat_data["messages"][-1]["role"], "user")

if __name__ == "__main__":
    unittest.main()
