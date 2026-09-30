import threading
import enum


class DialogState(enum.Enum):
    FREE = 1
    WAITING = 2
    SERVED = 3


class Dialog:
    def __init__(self):
        self.cond = threading.Condition()
        self.reply = ""
        self.text = ""
        self.state = DialogState.FREE

    @property
    def is_waiting(self):
        with self.cond:
            return self.state == DialogState.WAITING

    def get_text(self):
        with self.cond:
            return self.text

    def set_text(self, text):
        with self.cond:
            self.text = text
            self.cond.notify()

    def get_reply(self):
        with self.cond:
            reply = self.reply
            self.reply = ""
            return reply

    def set_reply(self, reply):
        with self.cond:
            self.reply = reply
            self.state = DialogState.SERVED
            self.cond.notify()

    def ask(self, text, cond):
        with self.cond:
            self.text = text
            self.state = DialogState.WAITING
            self.reply = ""
        with cond:
            cond.notify()
        with self.cond:
            while self.state != DialogState.SERVED:
                self.cond.wait()
            self.state = DialogState.FREE
            return self.reply
