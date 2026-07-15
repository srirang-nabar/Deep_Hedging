"""Stage 6 gate: write-up and verification pack.

Every number in the README's headline table and hypothesis summary must
equal a registered claim (numbers must match CLAIMS.md exactly — the claims
register is the single source of truth). The fresh-machine dry run must have
been recorded and passed. The report and interview QA must exist and quote
only registered values. Resume bullets must cite existing claim IDs.
"""

import re
from pathlib import Path

import pytest

import deep_hedging
from deep_hedging.claims import read_claims

pytestmark = pytest.mark.gate_stage6

PROJECT_ROOT = Path(deep_hedging.__file__).resolve().parents[2]
CLAIMS = read_claims()


def test_readme_headline_numbers_are_registered():
    """The learned-policy column and baseline cells of the README table are
    exactly the registered claim values (4 dp string match)."""
    readme = (PROJECT_ROOT / "README.md").read_text()
    for claim_id in (
        "LP5-CVAR95-MEAN", "LP20-CVAR95-MEAN", "LP50-CVAR95-MEAN",
        "BT20-DELTA-CVAR95", "BT20-LELAND-CVAR95", "BT20-WW-CVAR95",
        "BT50-DELTA-CVAR95", "BT50-LELAND-CVAR95", "BT50-WW-CVAR95",
        "H3-HESTON-CVAR-DIFF", "H3-NIFTY-CVAR-DIFF",
    ):
        value = CLAIMS[claim_id]["value"]
        rendered = f"{abs(value):.4f}"
        assert rendered in readme, f"README missing registered value {rendered} ({claim_id})"
    lo, hi = -0.2334, -0.2120  # H1 CI quoted in README must match HYPOTHESES.md
    hypotheses = (PROJECT_ROOT / "HYPOTHESES.md").read_text()
    assert f"[{lo:.4f}, {hi:.4f}]".replace("-", "−") in readme.replace("-", "−")
    assert f"{lo:.4f}".replace("-", "−") in hypotheses.replace("-", "−")


def test_fresh_machine_run_recorded_and_passed():
    log_path = PROJECT_ROOT / "results" / "fresh_machine_run.log"
    assert log_path.exists(), "run the fresh-machine dry run and record the transcript"
    log = log_path.read_text()
    assert "uv sync --frozen" in log
    assert "[tier1] PASS" in log
    assert "FAILED" not in log and "Traceback" not in log


def test_reproducing_has_measured_times():
    text = (PROJECT_ROOT / "REPRODUCING.md").read_text()
    assert "TIER1_MEASURED" not in text and "PLACEHOLDER" not in text
    assert "measured" in text.lower()


def test_report_quotes_only_registered_values():
    """Spot-check: the report's headline numbers exist in CLAIMS.md, and no
    placeholder tokens survived editing."""
    report = (PROJECT_ROOT / "report" / "report.md").read_text()
    assert "PLACEHOLDER" not in report
    for claim_id in ("LP20-CVAR95-MEAN", "BT50-LELAND-CVAR95", "H3-NIFTY-CVAR-DIFF"):
        assert f"{abs(CLAIMS[claim_id]['value']):.4f}" in report, claim_id


def test_resume_bullets_cite_existing_claims():
    bullets = (PROJECT_ROOT / "report" / "resume_bullets.md").read_text()
    cited = set(re.findall(r"[A-Z0-9]+-[A-Z0-9-]+", bullets))
    cited = {c for c in cited if any(c.startswith(p) for p in ("BT", "LP", "H1", "H2", "H3", "ABL"))}
    assert cited, "bullets must cite claim IDs"
    for claim_id in cited:
        if claim_id.endswith("-*"):
            prefix = claim_id[:-2]
            assert any(k.startswith(prefix) for k in CLAIMS), claim_id
        else:
            assert claim_id in CLAIMS, f"bullet cites unknown claim {claim_id}"


def test_interview_qa_exists_with_core_questions():
    qa = (PROJECT_ROOT / "report" / "interview_qa.md").read_text().lower()
    for topic in ("cvar", "leland", "differentiate through the sim", "straw", "misspecification", "volunteer"):
        assert topic in qa, f"interview QA missing topic: {topic}"
