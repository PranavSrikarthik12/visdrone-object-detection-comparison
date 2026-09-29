"""Re-apply importlib.resources patch to installed deimkit.

Replaces deprecated pkg_resources.resource_filename usage in
deimkit/config.py with stdlib importlib.resources.files + os.fspath.

Usage:
    python scripts/patch_deimkit.py
"""
import pathlib
import sys

OLD_IMPORT = "import pkg_resources"
NEW_IMPORT = "from importlib import resources as _resources"

OLD_USAGE = """            pkg_resources.resource_filename(
                "deimkit", f"configs/{model_name}_coco.yml"
            ),
            pkg_resources.resource_filename(
                "deimkit", f"configs/deim_dfine/{model_name}_coco.yml"
            ),"""

NEW_USAGE = """            os.fspath(
                _resources.files("deimkit").joinpath(
                    f"configs/{model_name}_coco.yml"
                )
            ),
            os.fspath(
                _resources.files("deimkit").joinpath(
                    f"configs/deim_dfine/{model_name}_coco.yml"
                )
            ),"""


def find_config() -> pathlib.Path:
    import importlib.util

    spec = importlib.util.find_spec("deimkit.config")
    if spec is None or spec.origin is None:
        print("deimkit not installed in current environment", file=sys.stderr)
        raise SystemExit(1)
    return pathlib.Path(spec.origin)


def main() -> None:
    path = find_config()
    text = path.read_text(encoding="utf-8")
    if "pkg_resources" not in text:
        print(f"already patched: {path}")
        return
    if OLD_IMPORT not in text or OLD_USAGE not in text:
        print(f"unexpected content in {path}, aborting", file=sys.stderr)
        raise SystemExit(2)
    text = text.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_USAGE, NEW_USAGE)
    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
