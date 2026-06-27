from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
ASSET_DIR = PACKAGE_ROOT / "assets"


def data_path(filename: str) -> Path:
    return DATA_DIR / filename


def asset_path(filename: str) -> Path:
    return ASSET_DIR / filename
