"""
MediScan — Root Application Entrypoint for Hugging Face Spaces & Streamlit Cloud.

Delegates directly to the canonical app.app.main() routine.
"""

from pathlib import Path
import sys

# Ensure repository root is on Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import main

if __name__ == "__main__":
    main()
