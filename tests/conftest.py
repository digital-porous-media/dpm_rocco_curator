"""Pytest configuration and fixtures.

This file is automatically loaded by pytest and ensures that the project
root is in the Python path so that 'src' can be imported.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
