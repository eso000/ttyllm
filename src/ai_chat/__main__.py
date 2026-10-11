"""Enable ``python -m ai_chat``.

Also works when this file is executed directly (``python src/ai_chat/``), in
which case there is no package context and the import has to be bootstrapped.
"""

if __package__:
    from .cli import main
else:  # pragma: no cover - only hit when run as a bare script
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ai_chat.cli import main

raise SystemExit(main())
