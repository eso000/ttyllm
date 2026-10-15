"""Tests for tab completion of commands, models and file paths."""

import os
import tempfile
import unittest

from ai_chat.completion import candidates, complete

MODELS = ["llama3.2", "llama3.1", "mistral"]


class TestCandidates(unittest.TestCase):
    """Tests for candidates."""

    def test_a_command_without_an_argument_has_none(self) -> None:
        """A command without an argument has none."""
        self.assertEqual(candidates(":sm", MODELS), ("", []))
        self.assertEqual(candidates("", MODELS), ("", []))

    def test_models_are_filtered_by_prefix(self) -> None:
        """Models are filtered by prefix."""
        self.assertEqual(candidates(":sm ll", MODELS), ("ll", ["llama3.2", "llama3.1"]))

    def test_an_empty_argument_offers_every_model(self) -> None:
        """An empty argument offers every model."""
        self.assertEqual(candidates(":sm ", MODELS)[1], MODELS)

    def test_both_prefixes_work(self) -> None:
        """Both prefixes work."""
        self.assertEqual(candidates("/sm mi", MODELS)[1], ["mistral"])

    def test_other_commands_offer_nothing(self) -> None:
        """Other commands offer nothing."""
        for command in (":o st", ":v un", ":bc x", ":nonsense x"):
            with self.subTest(command=command):
                self.assertEqual(candidates(command, MODELS)[1], [])

    def test_only_the_second_word_is_completed(self) -> None:
        """Only the second word is completed."""
        self.assertEqual(candidates(":sm ll extra", MODELS)[0], "ll")


class TestComplete(unittest.TestCase):
    """Tests for complete."""

    def test_single_match_completes_fully(self) -> None:
        """Single match completes fully."""
        self.assertEqual(complete(":sm mi", MODELS), "stral")

    def test_several_matches_complete_to_the_common_prefix(self) -> None:
        """Several matches complete to the common prefix."""
        self.assertEqual(complete(":sm ll", MODELS), "ama3.")

    def test_no_match_gives_nothing(self) -> None:
        """No match gives nothing."""
        self.assertEqual(complete(":sm zz", MODELS), "")

    def test_already_complete_gives_nothing(self) -> None:
        """Already complete gives nothing."""
        self.assertEqual(complete(":sm mistral", MODELS), "")

    def test_ambiguous_without_common_prefix_gives_nothing(self) -> None:
        """Ambiguous without common prefix gives nothing."""
        self.assertEqual(complete(":sm ", MODELS), "")

    def test_no_models(self) -> None:
        """No models."""
        self.assertEqual(complete(":sm l", []), "")


class TestPaths(unittest.TestCase):
    """Tests for paths."""

    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        for name in ("alpha.json", "alpine.json", "beta.json"):
            open(os.path.join(self.folder, name), "w", encoding="utf-8").close()
        os.mkdir(os.path.join(self.folder, "sub"))

    def test_files_complete_to_the_common_prefix(self) -> None:
        """Files complete to the common prefix."""
        for command in (":e", ":i", ":image", "/e", "/i", "/image"):
            with self.subTest(command=command):
                self.assertEqual(complete(f"{command} {self.folder}/al", []), "p")

    def test_a_single_file_completes_fully(self) -> None:
        """A single file completes fully."""
        self.assertEqual(complete(f":i {self.folder}/be", []), "ta.json")

    def test_a_directory_gets_a_trailing_slash(self) -> None:
        """A directory gets a trailing slash."""
        self.assertEqual(complete(f":i {self.folder}/su", []), "b/")

    def test_a_directory_with_a_slash_is_left_alone(self) -> None:
        """A directory with a slash is left alone."""
        self.assertEqual(complete(f":i {self.folder}/sub/", []), "")

    def test_nothing_matches(self) -> None:
        """Nothing matches."""
        self.assertEqual(complete(f":i {self.folder}/zzz", []), "")


if __name__ == "__main__":
    unittest.main()
