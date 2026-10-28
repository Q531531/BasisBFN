#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${PROJECT_DIR}/downloads}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/datasets/TransWeather}"

cd "${PROJECT_DIR}"
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi
command -v gdown >/dev/null || {
  echo "gdown is missing. Run scripts/setup_5090.sh first." >&2
  exit 1
}

mkdir -p "${DOWNLOAD_DIR}" "${DATASET_DIR}"
gdown --continue --fuzzy \
  "https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view" \
  -O "${DOWNLOAD_DIR}/all-weather.archive"
gdown --continue --fuzzy \
  "https://drive.google.com/file/d/1UsazX-P3sPcDGw3kxkyFWqUyNfhYN_AM/view" \
  -O "${DOWNLOAD_DIR}/all-weather-lists.archive"

DOWNLOAD_DIR="${DOWNLOAD_DIR}" DATASET_DIR="${DATASET_DIR}" python - <<'PY'
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

download_dir = Path(os.environ["DOWNLOAD_DIR"])
dataset_dir = Path(os.environ["DATASET_DIR"])


def validate_member(name: str) -> None:
    destination = (dataset_dir / name).resolve()
    root = dataset_dir.resolve()
    if destination != root and root not in destination.parents:
        raise RuntimeError(f"Unsafe archive path: {name}")


for archive in sorted(download_dir.glob("all-weather*.archive")):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                validate_member(member.filename)
            handle.extractall(dataset_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                validate_member(member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Archive links are not allowed: {member.name}")
            handle.extractall(dataset_dir)
    else:
        unpacked = dataset_dir / archive.stem
        shutil.copy2(archive, unpacked)
        print(f"Could not identify archive format; copied to {unpacked}")

manifests = sorted(dataset_dir.rglob("allweather.txt"))
if not manifests:
    raise SystemExit(
        "Download finished, but allweather.txt was not found. "
        "Inspect downloads/ and extract the provider archive manually."
    )
for manifest in manifests:
    data_root = manifest.parent.parent
    print(f"Found training manifest: {manifest}")
    print(f"Use: export SGB_DATA_ROOT={data_root}")
PY
