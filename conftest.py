"""
Root conftest. Configures pytest so that the two `tests/` packages
(top-level `tests/` and `api/tests/`) can be collected in the same run
without colliding on the shared `tests` module name.

This is only used during local development and CI. It has no effect on
the production API or App.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import src.*` and `import api.*`
# work regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
