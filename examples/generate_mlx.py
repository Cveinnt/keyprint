"""Compatibility entrypoint; prefer the installed keyprint generate command."""
from keyprint_v3.mlx_generate import main

if __name__ == "__main__":
    raise SystemExit(main())
