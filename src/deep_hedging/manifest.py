"""Artifact manifest tooling.

Responsibility: maintain results/MANIFEST.sha256 — one `<sha256>  <path>` line
per committed artifact (weights, frozen tables, data snapshots, path-set
definitions). `add` records or updates entries; `verify` recomputes every hash
and fails loudly on any mismatch, which is how volunteers check that the
artifacts they downloaded are the ones the claims were made from.

Usage:
    uv run python -m deep_hedging.manifest add results/baseline_table.json
    uv run python -m deep_hedging.manifest verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "results" / "MANIFEST.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Return {relative_path: sha256}. Missing manifest -> empty dict."""
    entries: dict[str, str] = {}
    if not manifest_path.exists():
        return entries
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel_path = line.split(maxsplit=1)
        entries[rel_path] = digest
    return entries


def write_manifest(entries: dict[str, str], manifest_path: Path = MANIFEST_PATH) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {rel_path}" for rel_path, digest in sorted(entries.items())]
    manifest_path.write_text("\n".join(lines) + "\n" if lines else "")


def add(
    paths: list[Path],
    manifest_path: Path = MANIFEST_PATH,
    *,
    allow_change: bool = True,
) -> None:
    """Hash each file and record/update its manifest entry.

    allow_change=False freezes existing entries: if a file's hash differs
    from the recorded one, raise instead of updating. Sign-off notebooks use
    this so that re-running them after a code change fails loudly rather
    than silently re-freezing a different benchmark.
    """
    entries = read_manifest(manifest_path)
    for path in paths:
        path = path.resolve()
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        digest = sha256_file(path)
        if not allow_change and rel_path in entries and entries[rel_path] != digest:
            raise RuntimeError(
                f"frozen artifact changed: {rel_path} hashes {digest[:16]}… but the "
                f"manifest records {entries[rel_path][:16]}…. If this change is "
                "intentional, update the entry explicitly with "
                "`uv run python -m deep_hedging.manifest add <path>`."
            )
        entries[rel_path] = digest
    write_manifest(entries, manifest_path)


def verify(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Recompute every hash; return a list of problems (empty = all good)."""
    problems = []
    entries = read_manifest(manifest_path)
    for rel_path, expected in entries.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            problems.append(f"missing: {rel_path}")
        elif sha256_file(path) != expected:
            problems.append(f"hash mismatch: {rel_path}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_add = sub.add_parser("add", help="hash files and record them in the manifest")
    p_add.add_argument("paths", nargs="+", type=Path)
    sub.add_parser("verify", help="recompute all hashes and report mismatches")
    args = parser.parse_args(argv)

    if args.command == "add":
        add(args.paths)
        return 0
    problems = verify()
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        print(f"manifest OK ({len(read_manifest())} artifacts)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
