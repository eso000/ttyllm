"""Run ai_chat straight from a checkout: ``python3 .`` (no install needed)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ai_chat.cli import main  # noqa: E402  (path must be set up first)

raise SystemExit(main())
