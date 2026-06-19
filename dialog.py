import threading


class Dialog:
    def __init__(self):
        self.cond = threading.Condition()
        self.reply = ""
        self.text = ""

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
            self.cond.notify()
