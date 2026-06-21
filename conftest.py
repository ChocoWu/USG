"""Ensures the repo root is on sys.path so `import usg_par` works under pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))