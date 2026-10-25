"""PyInstaller entry point (also permits command-line smoke testing)."""
from vertical_dynamics.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
