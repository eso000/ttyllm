from enum import Enum


class State(Enum):
    """which view is currently displayed"""
    NORMAL = 0
    SYSTEM = 1
    DATA = 2
    HELP = 3
    MODELS = 4
