"""
Veritas AI — Hybrid Fake News Detection & Verification Engine
Streamlit Application Entry Point.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is in python sys.path
sys_root = Path(__file__).resolve().parent
if str(sys_root) not in sys.path:
    sys.path.insert(0, str(sys_root))

from frontend.dashboard import main

if __name__ == "__main__":
    main()
