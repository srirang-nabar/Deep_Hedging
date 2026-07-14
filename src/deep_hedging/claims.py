"""Claims register tooling.

Responsibility: parse the CLAIMS.md table and cross-check it against freshly
computed values. Every results notebook ends with
verify_claims("<notebook_id>", computed) — it fails if a registered claim for
that notebook is missing from `computed`, if `computed` contains a number the
register doesn't know (register gone stale), or if any value disagrees beyond
`atol`. This is the mechanism behind the house rule: no number appears in the
README or a resume bullet unless a green assert backs it.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = PROJECT_ROOT / "CLAIMS.md"


def read_claims(path: Path = CLAIMS_PATH) -> dict[str, dict]:
    """Parse CLAIMS.md rows into {claim_id: {claim, value, notebook, test,
    stage}}. Non-numeric Value cells are skipped (prose-only claims)."""
    claims: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("ID", "") or set(cells[0]) <= {"-", " ", ":"}:
            continue
        try:
            value = float(cells[2])
        except ValueError:
            continue
        claims[cells[0]] = {
            "claim": cells[1],
            "value": value,
            "notebook": cells[3],
            "test": cells[4],
            "stage": cells[5],
        }
    return claims


def verify_claims(notebook_id: str, computed: dict[str, float], *, atol: float = 1e-4) -> str:
    """Cross-check every registered claim for `notebook_id` against the
    freshly computed values. Raises AssertionError listing every problem;
    returns a one-line summary when all claims verify."""
    registered = {k: v for k, v in read_claims().items() if v["notebook"] == notebook_id}
    problems = []
    for claim_id in sorted(set(registered) - set(computed)):
        problems.append(f"registered but not computed: {claim_id}")
    for claim_id in sorted(set(computed) - set(registered)):
        problems.append(f"computed but not registered (stale CLAIMS.md?): {claim_id}")
    for claim_id in sorted(set(registered) & set(computed)):
        want, got = registered[claim_id]["value"], computed[claim_id]
        if abs(want - got) > atol:
            problems.append(f"MISMATCH {claim_id}: register says {want}, computed {got:.6f}")
    assert not problems, "claims verification failed:\n  " + "\n  ".join(problems)
    return f"verified {len(registered)} claims for {notebook_id} (atol={atol})"
