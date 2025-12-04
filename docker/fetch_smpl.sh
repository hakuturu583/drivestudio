#!/usr/bin/env bash
# Fetch SMPL_NEUTRAL.pkl into /workspace/drivestudio/smpl_models during image build.
set -euo pipefail

export DEST="/workspace/drivestudio/smpl_models/SMPL_NEUTRAL.pkl"
export URL="${SMPL_NEUTRAL_URL:-https://smpl.is.tue.mpg.de/download.php?filename=SMPL_python_v.1.1.0.zip}"
export GDRIVE_ID="${SMPL_NEUTRAL_GDRIVE_ID:-}"
export STRICT="${SMPL_DOWNLOAD_STRICT:-1}"

python3 - <<'PY'
import os
import sys
import tempfile
import zipfile
import urllib.request

dest = os.environ.get("DEST", "/workspace/drivestudio/smpl_models/SMPL_NEUTRAL.pkl")
url = os.environ.get("URL", "")
gid = os.environ.get("GDRIVE_ID", "")
strict = os.environ.get("STRICT", "1") == "1"

def download_from_url(u: str) -> bool:
    if not u:
        return False
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    try:
        urllib.request.urlretrieve(u, tmp)
        return finalize_download(tmp)
    except Exception as exc:
        try:
            os.remove(tmp)
        except OSError:
            pass
        sys.stderr.write(f"[warn] URL download failed: {exc}\n")
        return False

def download_from_gdrive(file_id: str) -> bool:
    if not file_id:
        return False
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    try:
        import gdown
        gdown.download(id=file_id, output=tmp, quiet=False)
        return finalize_download(tmp)
    except Exception as exc:
        try:
            os.remove(tmp)
        except OSError:
            pass
        sys.stderr.write(f"[warn] gdown download failed: {exc}\n")
        return False

def finalize_download(tmp_path: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if zipfile.is_zipfile(tmp_path):
        with zipfile.ZipFile(tmp_path, "r") as zf:
            candidates = [
                name for name in zf.namelist()
                if name.lower().endswith("neutral_lbs_10_207_0_v1.1.0.pkl")
                or name.lower().endswith("smpl_neutral.pkl")
            ]
            if not candidates:
                raise RuntimeError("SMPL_NEUTRAL.pkl not found in zip")
            with zf.open(candidates[0], "r") as src, open(dest, "wb") as dst:
                dst.write(src.read())
        os.remove(tmp_path)
        return True
    os.replace(tmp_path, dest)
    return True

if os.path.exists(dest):
    print(f"SMPL model already present at {dest}")
    sys.exit(0)

source = None
if download_from_url(url):
    print(f"Downloaded SMPL_NEUTRAL.pkl from URL -> {dest}")
    source = "url"
elif download_from_gdrive(gid):
    print(f"Downloaded SMPL_NEUTRAL.pkl from Google Drive ID {gid} -> {dest}")
    source = "gdrive"

if source:
    try:
        import pickle
        with open(dest, "rb") as f:
            pickle.load(f, encoding="latin1")
        print(f"Validated SMPL pickle from {source}.")
        sys.exit(0)
    except Exception as exc:
        msg = f"Downloaded SMPL_NEUTRAL.pkl is invalid: {exc}"
        if strict:
            sys.stderr.write(msg + "\n")
            sys.exit(1)
        print("[warn]", msg)
        sys.exit(0)

msg = (
    "SMPL_NEUTRAL.pkl not downloaded. "
    "Provide SMPL_NEUTRAL_URL or SMPL_NEUTRAL_GDRIVE_ID (and set SMPL_DOWNLOAD_STRICT=0 to continue on failure)."
)
if strict:
    sys.stderr.write(msg + "\n")
    sys.exit(1)
print("[warn]", msg)
PY
