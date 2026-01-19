import unittest
import io
import sys

import ai_chat


class Capturer:
    def __init__(self):
        self.capt = None

    def begin_capture_stdout(self):
        self.capt = io.StringIO()
        sys.stdout = self.capt

    def end_capture_stdout(self) -> str:
        sys.stdout = sys.__stdout__
        return self.capt.getvalue()


class TestChat(unittest.TestCase):
    def test_getter_normal_color(self):
        sut = ai_chat.Chat()
        self.assertEqual(sut.normal_color, "\033[37m")

    def test_getter_user_color(self):
        sut = ai_chat.Chat()
        self.assertEqual(sut.user_color, "\033[36m")

    def test_getter_ai_color(self):
        sut = ai_chat.Chat()
        self.assertEqual(sut.ai_color, "\033[35m")

    def test_getter_user_name(self):
        sut = ai_chat.Chat()
        self.assertEqual(sut.user_name, "You")

    def test_getter_ai_name(self):
        sut = ai_chat.Chat()
        self.assertEqual(sut.ai_name, "AI")

    def test_setter_user_name(self):
        sut = ai_chat.Chat()
        sut.user_name = "TEST"
        self.assertEqual(sut.user_name, "TEST")

    def test_setter_ai_name(self):
        sut = ai_chat.Chat()
        sut.ai_name = "TEST"
        self.assertEqual(sut.ai_name, "TEST")

    def test_setter_normal_color(self):
        sut = ai_chat.Chat()
        sut.normal_color = "8b:1"
        self.assertEqual(sut.normal_color, "\033[31m")
        sut.normal_color = "256b:199"
        self.assertEqual(sut.normal_color, "\033[38;5;199m")
        sut.normal_color = "rgb:122;123;144"
        self.assertEqual(sut.normal_color, "\033[38;2;122;123;144m")
        with self.assertRaises(ai_chat.ColorCodeExeption) as err:
            sut.normal_color = "12"
        self.assertEqual(str(err.exception), "12 does not fit acceptable formats")

    def test_setter_user_color(self):
        sut = ai_chat.Chat()
        sut.user_color = "8b:1"
        self.assertEqual(sut.user_color, "\033[31m")
        sut.user_color = "256b:199"
        self.assertEqual(sut.user_color, "\033[38;5;199m")
        sut.user_color = "rgb:122;123;144"
        self.assertEqual(sut.user_color, "\033[38;2;122;123;144m")
        with self.assertRaises(ai_chat.ColorCodeExeption) as err:
            sut.user_color = "12"
        self.assertEqual(str(err.exception), "12 does not fit acceptable formats")

    def test_setter_ai_color(self):
        sut = ai_chat.Chat()
        sut.ai_color = "8b:1"
        self.assertEqual(sut.ai_color, "\033[31m")
        sut.ai_color = "256b:199"
        self.assertEqual(sut.ai_color, "\033[38;5;199m")
        sut.ai_color = "rgb:122;123;144"
        self.assertEqual(sut.ai_color, "\033[38;2;122;123;144m")
        with self.assertRaises(ai_chat.ColorCodeExeption) as err:
            sut.ai_color = "12"
        self.assertEqual(str(err.exception), "12 does not fit acceptable formats")


class TestPrint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_chat = ai_chat.Chat()
        cls.test_chat.chat_data["messages"] = [
            {"role": "system", "content": "green"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "yellow"},
        ]
        cls._expected_result_print_messages = (
            "\033[H\033[J\033[3J"
            "\033[37m0\n"
            "\033[37m{'role': 'system', 'content': 'green'}\n"
            "\033[37m1\n"
            "\033[37m{'role': 'user', 'content': 'hello'}\n"
            "\033[37m2\n"
            "\033[37m{'role': 'assistant', 'content': 'yellow'}\n"
        )
        cls._expected_result_reprint = (
            "\033[H\033[J\033[3J"
            "\033[37mYou:\n"
            "\033[36mhello\n"
            "\033[37mAI:\n"
            "\033[35myellow\n"
        )
        cls._expected_result_print_data = (
            "\033[37mmodel: mistral-nemo\n"
            "\033[37mstream: True\n"
            "\033[37mthink: False\n"
        )

    def test_clean_screen(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.clean_screan()
        self.assertEqual(cap.end_capture_stdout(), "\033[H\033[J\033[3J")

    def test_print_user(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.print_user(self.test_chat, "TEST")
        self.assertEqual(cap.end_capture_stdout(), "\033[36mTEST\n")

    def test_print_normal(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.print_normal(self.test_chat, "TEST")
        self.assertEqual(cap.end_capture_stdout(), "\033[37mTEST\n")

    def test_print_ai(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.print_ai(self.test_chat, "TEST")
        self.assertEqual(cap.end_capture_stdout(), "\033[35mTEST\n")

    def test_print_messages(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.print_messages(self.test_chat)
        self.assertEqual(cap.end_capture_stdout(), self._expected_result_print_messages)

    def test_print_reprint(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.reprint(self.test_chat)
        self.assertEqual(cap.end_capture_stdout(), self._expected_result_reprint)

    def test_print_print_data(self):
        cap = Capturer()
        cap.begin_capture_stdout()
        ai_chat.Print.print_data(self.test_chat)
        self.assertEqual(cap.end_capture_stdout(), self._expected_result_print_data)


class TestChatList(unittest.TestCase):
    def test_get_current_chat(self):
        sut = ai_chat.ChatList()
        self.assertEqual(sut.current_chat, sut.chats[sut.current_chat_index])

    def test_get_current_data(self):
        sut = ai_chat.ChatList()
        self.assertEqual(sut.current_data, sut.chats[sut.current_chat_index].chat_data)

    def test_creae_new_chat(self):
        sut = ai_chat.ChatList()
        self.assertEqual(sut.current_chat_index, 0)
        self.assertEqual(len(sut.chats), 1)
        sut.creat_new_chat()
        self.assertEqual(len(sut.chats), 2)
        self.assertEqual(sut.current_chat_index, 1)

    def test_previous_chat(self):
        sut = ai_chat.ChatList()
        sut.creat_new_chat()
        self.assertEqual(sut.current_chat_index, 1)
        sut.previous_chat()
        self.assertEqual(sut.current_chat_index, 0)

    def test_next_chat(self):
        sut = ai_chat.ChatList()
        sut.creat_new_chat()
        sut.previous_chat()
        self.assertEqual(sut.current_chat_index, 0)
        sut.next_chat()
        self.assertEqual(sut.current_chat_index, 1)

    def test_delete_current_chat(self):
        sut = ai_chat.ChatList()
        sut.creat_new_chat()
        self.assertEqual(sut.current_chat_index, 1)
        self.assertEqual(len(sut.chats), 2)
        sut.delete_current_chat()
        self.assertEqual(sut.current_chat_index, 0)
        self.assertEqual(len(sut.chats), 1)


if __name__ == "__main__":
    unittest.main()
