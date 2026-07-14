"""Stage 0 gate: the environment and protocol skeleton is in place."""

import importlib
from pathlib import Path

import pytest

import deep_hedging
from deep_hedging import manifest

PROJECT_ROOT = Path(deep_hedging.__file__).resolve().parents[2]

MODULES = [
    "deep_hedging.simulate",
    "deep_hedging.pricing",
    "deep_hedging.baselines",
    "deep_hedging.evaluate",
    "deep_hedging.policy",
    "deep_hedging.train",
    "deep_hedging.reproduce",
    "deep_hedging.manifest",
]

PROTOCOL_FILES = ["REPRODUCING.md", "CLAIMS.md", "HYPOTHESES.md", "README.md", "uv.lock"]


@pytest.mark.gate_stage0
@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_with_docstring(module_name):
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} must carry a responsibility docstring"


@pytest.mark.gate_stage0
@pytest.mark.parametrize("filename", PROTOCOL_FILES)
def test_protocol_file_exists(filename):
    assert (PROJECT_ROOT / filename).exists()


@pytest.mark.gate_stage0
def test_manifest_round_trip(tmp_path):
    artifact = PROJECT_ROOT / "results" / "_stage0_probe.txt"
    artifact.parent.mkdir(exist_ok=True)  # git does not track empty dirs
    artifact.write_text("probe\n")
    manifest_path = tmp_path / "MANIFEST.sha256"
    try:
        manifest.add([artifact], manifest_path)
        assert manifest.verify(manifest_path) == []

        artifact.write_text("tampered\n")
        problems = manifest.verify(manifest_path)
        assert problems and "mismatch" in problems[0]

        with pytest.raises(RuntimeError, match="frozen artifact changed"):
            manifest.add([artifact], manifest_path, allow_change=False)
        manifest.add([artifact], manifest_path)  # explicit update allowed
        assert manifest.verify(manifest_path) == []
    finally:
        artifact.unlink()
    problems = manifest.verify(manifest_path)
    assert problems and "missing" in problems[0]
