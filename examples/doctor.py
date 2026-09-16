"""Read-only setup check. Installs nothing and never loads a model."""
import argparse
import importlib.metadata
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx", action="store_true", help="also check the optional local-model environment")
    args = parser.parse_args()
    failures = []
    print(f"Python: {platform.python_version()} ({platform.system()} {platform.machine()})")
    if sys.version_info < (3, 12):
        failures.append("Python 3.12+ required")
    packages = ["keyprint-research-v3", "numpy", "scipy", "tokenizers"]
    if args.mlx:
        packages += ["mlx", "mlx-lm", "transformers"]
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            failures.append("generate_mlx.py requires Apple Silicon macOS")
    for name in packages:
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing package: {name}")
    if not failures:
        for command in ([sys.executable, "-m", "pip", "check"],
                        [sys.executable, "-m", "keyprint_v3", "verify"]):
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode:
                failures.append(result.stdout.strip() + "\n" + result.stderr.strip())
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if not failures:
        print("PASS: dependencies and SDK bundle verified. Model execution has not been tested.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
