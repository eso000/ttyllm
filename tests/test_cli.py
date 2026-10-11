import unittest
from unittest import mock

from ai_chat import cli
from ai_chat.requester import Requester
from ai_chat.supervisor import Supervisor


class TestParser(unittest.TestCase):
    def test_endpoint_defaults_to_none(self) -> None:
        self.assertIsNone(cli.build_parser().parse_args([]).endpoint)

    def test_endpoint_is_parsed(self) -> None:
        args = cli.build_parser().parse_args(["--endpoint", "example.com:1234"])
        self.assertEqual(args.endpoint, "example.com:1234")

    def test_short_endpoint_flag(self) -> None:
        args = cli.build_parser().parse_args(["-H", "https://api.example.com"])
        self.assertEqual(args.endpoint, "https://api.example.com")

    def test_insecure_flag_defaults_off(self) -> None:
        self.assertFalse(cli.build_parser().parse_args([]).insecure)

    def test_insecure_flags_are_parsed(self) -> None:
        self.assertTrue(cli.build_parser().parse_args(["-k"]).insecure)
        self.assertTrue(cli.build_parser().parse_args(["--insecure"]).insecure)


class TestMain(unittest.TestCase):
    def test_options_are_passed_to_the_supervisor(self) -> None:
        with mock.patch.object(
            Supervisor, "__init__", return_value=None
        ) as init, mock.patch.object(Supervisor, "run_chat"):
            cli.main(["--endpoint", "https://local:99", "--insecure"])
        init.assert_called_once_with(
            open_api=False, endpoint="https://local:99", insecure=True
        )

    def test_endpoint_is_applied_before_the_chat_runs(self) -> None:
        with mock.patch.object(Requester, "set_endpoint") as set_endpoint, mock.patch.object(
            Supervisor, "run_chat"
        ) as run_chat:
            cli.main(["--endpoint", "example.com:1234"])
        set_endpoint.assert_called_once_with("example.com:1234")
        run_chat.assert_called_once_with()

    def test_without_endpoint_nothing_is_changed(self) -> None:
        with mock.patch.object(Requester, "set_endpoint") as set_endpoint, mock.patch.object(
            Supervisor, "run_chat"
        ):
            cli.main([])
        set_endpoint.assert_not_called()

    def test_invalid_endpoint_exits_with_a_usage_error(self) -> None:
        with mock.patch.object(Supervisor, "run_chat"):
            with self.assertRaises(SystemExit) as context:
                cli.main(["--endpoint", "host:notaport"])
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
