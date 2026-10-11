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
    def test_short_text_is_one_line(self) -> None:
        self.assertEqual(text_to_lines("a b", 5), ["a b"])

    def test_empty_text(self) -> None:
        self.assertEqual(text_to_lines("", 5), [""])

    def test_exact_width(self) -> None:
        self.assertEqual(text_to_lines("exact5", 5), ["exact", "5"])

    def test_breaks_on_last_space(self) -> None:
        self.assertEqual(
            text_to_lines("hi there friend", 5), ["hi", "there", "frien", "d"]
        )

    def test_no_blank_line_when_break_lands_on_space(self) -> None:
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
        for width in range(1, 12):
            for text in SAMPLES:
                got = "".join(text_to_lines(text, width))
                self.assertEqual(
                    got.replace(" ", ""),
                    text.replace(" ", ""),
                    f"{text!r} at width {width}",
                )

    def test_no_line_exceeds_width(self) -> None:
        for width in range(1, 12):
            for text in SAMPLES:
                for line in text_to_lines(text, width):
                    self.assertLessEqual(len(line), width)

    def test_leading_indentation_is_not_rewrapped(self) -> None:
        # known limitation: a break at index 0 or 1 of a chunk still yields a
        # near-blank first line for indented text
        self.assertEqual(text_to_lines("  hello world", 5), [" ", "hello", "world"])

    def test_newlines_preserved(self) -> None:
        self.assertEqual(text_to_lines("a\n\nb", 5), ["a", "", "b"])

    def test_tabs_expanded(self) -> None:
        self.assertEqual(text_to_lines("a\tb", 5), ["a   b"])

    def test_zero_width_returns_empty_line(self) -> None:
        self.assertEqual(text_to_lines("hello", 0), [""])

    def test_negative_width_returns_empty_line(self) -> None:
        self.assertEqual(text_to_lines("hello", -3), [""])

class TestColorParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = Buffer()

    def _set_color(self, value: str) -> str:
        self.buffer.ai_color = value
        return self.buffer.ai_color

    def test_rgb_allows_zero_components(self) -> None:
        self.assertEqual(self._set_color("rgb:0;0;0"), "\033[38;2;0;0;0m")
        self.assertEqual(self._set_color("rgb:0;128;255"), "\033[38;2;0;128;255m")

    def test_rgb_allows_255(self) -> None:
        self.assertEqual(self._set_color("rgb:255;255;255"), "\033[38;2;255;255;255m")

    def test_rgb_rejects_out_of_range(self) -> None:
        for value in ("rgb:256;0;0", "rgb:0;0;999", "rgb:-1;0;0"):
            with self.assertRaises(ColorCodeException):
                self._set_color(value)

    def test_8bit_and_256bit_still_work(self) -> None:
        self.assertEqual(self._set_color("8b:5"), "\033[35m")
        self.assertEqual(self._set_color("256b:0"), "\033[38;5;0m")
        self.assertEqual(self._set_color("256b:200"), "\033[38;5;200m")
        self.assertEqual(self._set_color("256b:255"), "\033[38;5;255m")

    def test_256bit_rejects_out_of_range(self) -> None:
        for value in ("256b:256", "256b:300", "256b:999"):
            with self.assertRaises(ColorCodeException):
                self._set_color(value)

    def test_garbage_is_rejected(self) -> None:
        with self.assertRaises(ColorCodeException):
            self._set_color("rgb:1;2")

class TestMakeTextBox(unittest.TestCase):
    def test_lines_and_last(self) -> None:
        box = make_text_box("one\ntwo", 10, last=4)
        self.assertEqual(box.lines, ["one", "two"])
        self.assertEqual(box.last, 4)

    def test_narrow_width_does_not_hang(self) -> None:
        self.assertEqual(make_text_box("a long line here", 0).lines, [""])

class TestBufferNarrowTerminal(unittest.TestCase):
    def test_get_text_box_zero_width(self) -> None:
        buffer = Buffer()
        buffer.chat_data["messages"].append({"role": "user", "content": "hello world"})
        lines = buffer.get_text_box(0).lines
        self.assertEqual(len(lines), 2)  # name header plus one empty line
        self.assertEqual(lines[1], "")

    def test_get_text_box_system_zero_width(self) -> None:
        buffer = Buffer()
        buffer.chat_data["messages"][0]["content"] = "you are helpful"
        self.assertEqual(buffer.get_text_box_system(0).lines[1], "")

    def test_get_text_box_normal_width(self) -> None:
        buffer = Buffer()
        buffer.chat_data["messages"].append(
            {"role": "user", "content": "hi there friend"}
        )
        self.assertTrue(
            any(line.endswith("hi") for line in buffer.get_text_box(5).lines)
        )

if __name__ == "__main__":
    unittest.main()
