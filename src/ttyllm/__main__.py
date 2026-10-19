"""Enable ``python -m ttyllm``.

Also works when this file is executed directly (``python src/ttyllm/``), in
which case there is no package context and the import has to be bootstrapped.
"""

if __package__:
    from .cli import main
else:  # pragma: no cover - only hit when run as a bare script
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ttyllm.cli import main

raise SystemExit(main())
