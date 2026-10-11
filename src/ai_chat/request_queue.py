import threading

from .buffer import Buffer


class RequestQueue:
    """serialises reply requests, one buffer at a time"""
    def __init__(self) -> None:
        """create an empty queue"""
        self.cond: threading.Condition = threading.Condition()
        self.qu: list[Buffer] = []

    def schedule(self, chat: Buffer) -> None:
        """add a chat to the queue unless it is already queued or running"""
        with self.cond:
            if chat.queued or chat.in_request:
                return
            self.qu.append(chat)
            chat.queued = True
            self.cond.notify()
            chat.notify()

    def deschedule(self, chat: Buffer) -> None:
        """remove a chat from the queue"""
        with self.cond:
            for index, scheduled_chat in enumerate(self.qu):
                if scheduled_chat == chat:
                    chat.queued = False
                    self.qu.pop(index)
                    chat.notify()
                    return

    @property
    def next(self) -> Buffer:
        """pop the oldest queued chat"""
        with self.cond:
            chat = self.qu.pop(0)
            chat.queued = False
            chat.notify()
            return chat

    @property
    def has_scheduled(self) -> bool:
        """True when at least one chat waits for a reply"""
        with self.cond:
            return len(self.qu) != 0
