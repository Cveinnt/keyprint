"""Installation check. Prefer `keyprint doctor` after installation."""
from keyprint.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["doctor"]))
