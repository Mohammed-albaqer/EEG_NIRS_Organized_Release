"""Download public competition .mat sessions from Yandex Object Storage.

Examples:
    python download_sessions.py --out data_all
    python download_sessions.py --out data_all --subjects N01 N02 N03
    python download_sessions.py --out data_all --exclude-subjects N17 N18

Existing complete files are skipped.  The script uses only the Python standard
library so it works before installing numpy/scipy/sklearn.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://storage.yandexcloud.net/neuralinterfaces-train/"


def list_remote_sessions() -> list[tuple[str, int]]:
    req = urllib.request.Request(BUCKET, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        xml_data = r.read()
    root = ET.fromstring(xml_data)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out: list[tuple[str, int]] = []
    for content in root.findall("s3:Contents", ns):
        key = content.findtext("s3:Key", default="", namespaces=ns)
        size_s = content.findtext("s3:Size", default="0", namespaces=ns)
        if key.startswith("Data/") and key.endswith(".mat") and "_N" in key:
            out.append((key, int(size_s)))
    return sorted(out)


def subject_from_key(key: str) -> str:
    return Path(key).stem.split("_")[-1].upper()


def download_one(key: str, size: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(key).name
    if dest.exists() and dest.stat().st_size == size:
        print(f"SKIP {dest.name} ({size/1024/1024:.1f} MiB already complete)")
        return

    url = BUCKET + urllib.parse.quote(key, safe="/")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"GET  {dest.name} ({size/1024/1024:.1f} MiB)")
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                copied = 0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    copied += len(chunk)
                    print(f"     {100.0*copied/max(size,1):5.1f}%", end="\r", flush=True)
            print(" " * 24, end="\r")
            if size and tmp.stat().st_size != size:
                raise IOError(f"size mismatch: got {tmp.stat().st_size}, expected {size}")
            tmp.replace(dest)
            print(f"OK   {dest.name}")
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == 4:
                raise
            print(f"Retry {attempt}/3 after download error: {exc}")
            time.sleep(min(2 ** attempt, 8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data_all")
    ap.add_argument("--subjects", nargs="*", default=[])
    ap.add_argument("--exclude-subjects", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="0 = all selected sessions")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    sessions = list_remote_sessions()
    include = {s.upper() for s in args.subjects}
    exclude = {s.upper() for s in args.exclude_subjects}
    if include:
        sessions = [x for x in sessions if subject_from_key(x[0]) in include]
    if exclude:
        sessions = [x for x in sessions if subject_from_key(x[0]) not in exclude]
    if args.limit > 0:
        sessions = sessions[: args.limit]

    total = sum(size for _, size in sessions)
    print(f"Selected {len(sessions)} sessions, total {total/1024/1024/1024:.2f} GiB")
    for key, size in sessions:
        print(f"  {subject_from_key(key):>4}  {size/1024/1024:6.1f} MiB  {Path(key).name}")
    if args.list_only:
        return 0

    out = Path(args.out)
    for i, (key, size) in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}]")
        try:
            download_one(key, size, out)
        except Exception as exc:
            print(f"ERROR downloading {key}: {exc}", file=sys.stderr)
            return 2
    print(f"Done. Files are in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
