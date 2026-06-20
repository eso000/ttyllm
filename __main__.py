import sys

from supervisor import Supervisor
from buffer import Buffer

if __name__ == "__main__":
    if "-t" in sys.argv:
        Buffer.TOOLS = True
    Supervisor(open_api="-o" in sys.argv).run_chat()
