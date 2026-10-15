"""Tests for text wrapping, color parsing and box layout."""

import unittest

from ai_chat.box_drawing import make_text_box, text_to_lines
from ai_chat.buffer import Buffer, ColorCodeException

SAMPLES: list[str] = [
    "hi there friend",
    "a b c d e f g",
    "one  two   three",
    "exact5",
    "aaaa bb",
    "a" * 40,
    ("word " * 20).strip(),
]

# indentation and trailing blanks are not wrapped cleanly by the greedy
# last-space break; see test_leading_indentation_is_not_rewrapped
EDGE_SAMPLES: list[str] = [
    "  hello world",
    "trailing space ",
    " leading",
]


class TestTextToLines(unittest.TestCase):
    """Tests for text to lines."""

    def test_short_text_is_one_line(self) -> None:
        """Short text is one line."""
        self.assertEqual(text_to_lines("a b", 5), ["a b"])

    def test_empty_text(self) -> None:
        """Empty text."""
        self.assertEqual(text_to_lines("", 5), [""])

    def test_exact_width(self) -> None:
        """Exact width."""
        self.assertEqual(text_to_lines("exact5", 5), ["exact", "5"])

    def test_breaks_on_last_space(self) -> None:
        """Breaks on last space."""
        self.assertEqual(
            text_to_lines("hi there friend", 5), ["hi", "there", "frien", "d"]
        )

    def test_no_blank_line_when_break_lands_on_space(self) -> None:
        """No blank line when break lands on space."""
        for width in range(2, 12):
            for text in SAMPLES:
                for line in text_to_lines(text, width):
                    self.assertNotEqual(
                        line.strip(),
                        "",
                        f"blank line from {text!r} at width {width}",
                    )

    def test_no_text_is_lost(self) -> None:
        # spaces at a wrap point are consumed as the break
        """No text is lost."""
        for width in range(1, 12):
            for text in SAMPLES:
                got = "".join(text_to_lines(text, width))
                self.assertEqual(
                    got.replace(" ", ""),
                    text.replace(" ", ""),
                    f"{text!r} at width {width}",
                )

    def test_no_line_exceeds_width(self) -> None:
        """No line exceeds width."""
        for width in range(1, 12):
            for text in SAMPLES:
                for line in text_to_lines(text, width):
                    self.assertLessEqual(len(line), width)

    def test_leading_indentation_is_not_rewrapped(self) -> None:
        # known limitation: a break at index 0 or 1 of a chunk still yields a
        # near-blank first line for indented text
        """Leading indentation is not rewrapped."""
        self.assertEqual(text_to_lines("  hello world", 5), [" ", "hello", "world"])

    def test_newlines_preserved(self) -> None:
        """Newlines preserved."""
        self.assertEqual(text_to_lines("a\n\nb", 5), ["a", "", "b"])

    def test_tabs_expanded(self) -> None:
        """Tabs expanded."""
        self.assertEqual(text_to_lines("a\tb", 5), ["a   b"])

    def test_zero_width_returns_empty_line(self) -> None:
        """Zero width returns empty line."""
        self.assertEqual(text_to_lines("hello", 0), [""])

    def test_negative_width_returns_empty_line(self) -> None:
        """Negative width returns empty line."""
        self.assertEqual(text_to_lines("hello", -3), [""])


class TestColorParsing(unittest.TestCase):
    """Tests for color parsing."""

    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        self.buffer = Buffer()

    def _set_color(self, value: str) -> str:
        """Test helper: set color."""
        self.buffer.ai_color = value
        return self.buffer.ai_color

    def test_rgb_allows_zero_components(self) -> None:
        """Rgb allows zero components."""
        self.assertEqual(self._set_color("rgb:0;0;0"), "\033[38;2;0;0;0m")
        self.assertEqual(self._set_color("rgb:0;128;255"), "\033[38;2;0;128;255m")

    def test_rgb_allows_255(self) -> None:
        """Rgb allows 255."""
        self.assertEqual(self._set_color("rgb:255;255;255"), "\033[38;2;255;255;255m")

    def test_rgb_rejects_out_of_range(self) -> None:
        """Rgb rejects out of range."""
        for value in ("rgb:256;0;0", "rgb:0;0;999", "rgb:-1;0;0"):
            with self.assertRaises(ColorCodeException):
                self._set_color(value)

    def test_8bit_and_256bit_still_work(self) -> None:
        """8bit and 256bit still work."""
        self.assertEqual(self._set_color("8b:5"), "\033[35m")
        self.assertEqual(self._set_color("256b:0"), "\033[38;5;0m")
        self.assertEqual(self._set_color("256b:200"), "\033[38;5;200m")
        self.assertEqual(self._set_color("256b:255"), "\033[38;5;255m")

    def test_256bit_rejects_out_of_range(self) -> None:
        """256bit rejects out of range."""
        for value in ("256b:256", "256b:300", "256b:999"):
            with self.assertRaises(ColorCodeException):
                self._set_color(value)

    def test_garbage_is_rejected(self) -> None:
        """Garbage is rejected."""
        with self.assertRaises(ColorCodeException):
            self._set_color("rgb:1;2")


class TestMakeTextBox(unittest.TestCase):
    """Tests for make text box."""

    def test_lines_and_last(self) -> None:
        """Lines and last."""
        box = make_text_box("one\ntwo", 10, last=4)
        self.assertEqual(box.lines, ["one", "two"])
        self.assertEqual(box.last, 4)

    def test_narrow_width_does_not_hang(self) -> None:
        """Narrow width does not hang."""
        self.assertEqual(make_text_box("a long line here", 0).lines, [""])


class TestBufferNarrowTerminal(unittest.TestCase):
    """Tests for buffer narrow terminal."""

    def test_get_text_box_zero_width(self) -> None:
        """Get text box zero width."""
        buffer = Buffer()
        buffer.chat_data["messages"].append({"role": "user", "content": "hello world"})
        lines = buffer.get_text_box(0).lines
        self.assertEqual(len(lines), 2)  # name header plus one empty line
        self.assertEqual(lines[1], "")

    def test_get_text_box_system_zero_width(self) -> None:
        """Get text box system zero width."""
        buffer = Buffer()
        buffer.chat_data["messages"][0]["content"] = "you are helpful"
        self.assertEqual(buffer.get_text_box_system(0).lines[1], "")

    def test_get_text_box_normal_width(self) -> None:
        """Get text box normal width."""
        buffer = Buffer()
        buffer.chat_data["messages"].append(
            {"role": "user", "content": "hi there friend"}
        )
        self.assertTrue(
            any(line.endswith("hi") for line in buffer.get_text_box(5).lines)
        )


if __name__ == "__main__":
    unittest.main()
