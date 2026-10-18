#!/usr/bin/env python3
"""Compatibility CLI alias for the canonical runtime/client.py implementation.

Usage remains: python action_aware_grpo/windows_client.py [client arguments].
The runtime directory is added explicitly so direct-script and module execution
resolve the same simulator service and pose helpers without relying on cwd.
"""
from pathlib import Path
import runpy
import sys


def main():
    runtime_root = Path(__file__).resolve().parent / "runtime"
    sys.path.insert(0, str(runtime_root))
    runpy.run_path(str(runtime_root / "client.py"), run_name="__main__")


if __name__ == "__main__":
    main()
