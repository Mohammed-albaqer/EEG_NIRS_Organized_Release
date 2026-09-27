"""Check the full public dataset against the downloaded file manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader((HERE / "public_data_manifest.csv").open(encoding="utf-8", newline="")))
    if len(rows) != 56:
        raise RuntimeError(f"Manifest should contain 56 sessions, found {len(rows)}")
    bad = []
    for row in rows:
        path = args.data / row["session"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            bad.append(row["session"])
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            bad.append(row["session"])
    if bad:
        raise RuntimeError("Missing or different sessions: " + ", ".join(bad))
    print("PASS: all 56 public session files match their SHA256 and byte lengths")


if __name__ == "__main__":
    main()
