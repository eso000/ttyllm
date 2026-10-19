"""Tests for the command-line argument parser and main()."""

import unittest
from unittest import mock

from ttyllm import cli
from ttyllm.requester import Requester
from ttyllm.supervisor import Supervisor
from ttyllm.api_key import ApiKey, ApiKeyType


class TestParser(unittest.TestCase):
    """Tests for parser."""

    def test_endpoint_defaults_to_none(self) -> None:
        """Endpoint defaults to none."""
        self.assertIsNone(cli.build_parser().parse_args([]).endpoint)

    def test_endpoint_is_parsed(self) -> None:
        """Endpoint is parsed."""
        args = cli.build_parser().parse_args(["--endpoint", "example.com:1234"])
        self.assertEqual(args.endpoint, "example.com:1234")

    def test_short_endpoint_flag(self) -> None:
        """Short endpoint flag."""
        args = cli.build_parser().parse_args(["-H", "https://api.example.com"])
        self.assertEqual(args.endpoint, "https://api.example.com")

    def test_insecure_flag_defaults_off(self) -> None:
        """Insecure flag defaults off."""
        self.assertFalse(cli.build_parser().parse_args([]).insecure)

    def test_insecure_flags_are_parsed(self) -> None:
        """Insecure flags are parsed."""
        self.assertTrue(cli.build_parser().parse_args(["-k"]).insecure)
        self.assertTrue(cli.build_parser().parse_args(["--insecure"]).insecure)

    def test_api_key_flags_are_parsed(self) -> None:
        """Api key flags are parsed."""
        args = cli.build_parser().parse_args(["--api-key", "sk-123"])
        self.assertEqual(args.api_key, "sk-123")
        args = cli.build_parser().parse_args(["--api-key-env", "MY_KEY"])
        self.assertEqual(args.api_key_env, "MY_KEY")
        args = cli.build_parser().parse_args(["--api-key-file", "/tmp/key"])
        self.assertEqual(args.api_key_file, "/tmp/key")

    def test_api_key_flags_default_off(self) -> None:
        """Api key flags default off."""
        args = cli.build_parser().parse_args([])
        self.assertIsNone(args.api_key)
        self.assertIsNone(args.api_key_env)
        self.assertIsNone(args.api_key_file)

    def test_api_key_flags_are_mutually_exclusive(self) -> None:
        """Api key flags are mutually exclusive."""
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["--api-key", "a", "--api-key-env", "B"])


class TestMain(unittest.TestCase):
    """Tests for main."""

    def test_options_are_passed_to_the_supervisor(self) -> None:
        """Options are passed to the supervisor."""
        with (
            mock.patch.object(Supervisor, "__init__", return_value=None) as init,
            mock.patch.object(Supervisor, "run_chat"),
        ):
            cli.main(["--endpoint", "https://local:99", "--insecure"])
        init.assert_called_once_with(
            open_api=False,
            endpoint="https://local:99",
            insecure=True,
            api_key=None,
        )

    def test_api_key_is_passed_to_the_supervisor(self) -> None:
        """Api key is passed to the supervisor."""
        with (
            mock.patch.object(Supervisor, "__init__", return_value=None) as init,
            mock.patch.object(Supervisor, "run_chat"),
        ):
            cli.main(["--api-key", "sk-123"])
        init.assert_called_once_with(
            open_api=False,
            endpoint=None,
            insecure=False,
            api_key=ApiKey("sk-123", ApiKeyType.RAW),
        )

    def test_endpoint_is_applied_before_the_chat_runs(self) -> None:
        """Endpoint is applied before the chat runs."""
        with (
            mock.patch.object(Requester, "set_endpoint") as set_endpoint,
            mock.patch.object(Supervisor, "run_chat") as run_chat,
        ):
            cli.main(["--endpoint", "example.com:1234"])
        set_endpoint.assert_called_once_with("example.com:1234")
        run_chat.assert_called_once_with()

    def test_without_endpoint_nothing_is_changed(self) -> None:
        """Without endpoint nothing is changed."""
        with (
            mock.patch.object(Requester, "set_endpoint") as set_endpoint,
            mock.patch.object(Supervisor, "run_chat"),
        ):
            cli.main([])
        set_endpoint.assert_not_called()

    def test_invalid_endpoint_exits_with_a_usage_error(self) -> None:
        """Invalid endpoint exits with a usage error."""
        with mock.patch.object(Supervisor, "run_chat"):
            with self.assertRaises(SystemExit) as context:
                cli.main(["--endpoint", "host:notaport"])
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
