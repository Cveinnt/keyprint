"""CLI shortcut for the pinned MLX backend."""
import sys
from keyprint.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", "--backend", "mlx", *sys.argv[1:]]))
