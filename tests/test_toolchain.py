import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolchain import read_file


class TestReadFile(unittest.TestCase):
    def test_reads_utf8_content(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False
        ) as file:
            file.write("héllo — 世界\n")
            path = file.name
        try:
            self.assertEqual(read_file(path), "héllo — 世界\n")
        finally:
            os.unlink(path)

    def test_reads_utf8_bytes_regardless_of_locale(self) -> None:
        data = "naïve café".encode("utf-8")
        with tempfile.NamedTemporaryFile("wb", delete=False) as file:
            file.write(data)
            path = file.name
        try:
            self.assertEqual(read_file(path), "naïve café")
        finally:
            os.unlink(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_file("/nonexistent/path/definitely-not-here")


if __name__ == "__main__":
    unittest.main()
