"""Smoke tests — verify the project layout, importability of core modules, and metric integrity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_layout():
    for p in ("src/core", "src/evaluation", "experiments/results", "plots", "figures", "config"):
        assert (ROOT / p).exists(), f"missing {p}"


def test_core_modules_present():
    for mod in (
        "retrieval_decision.py",
        "retrieval_engine.py",
        "grounding.py",
        "refinement.py",
        "answer_generation.py",
        "agent_controller.py",
    ):
        assert (ROOT / "src" / "core" / mod).is_file(), f"missing core/{mod}"


def test_ablation_results_structure():
    path = ROOT / "experiments" / "results" / "ablation_results_1000.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "results" in data
    assert "full_system" in data["results"]
    full = data["results"]["full_system"]["metrics"]
    assert 0 <= full["hallucination_rate"] <= 1
    assert 0 <= full["f1_score"] <= 1
    # The headline claim: full system is at least as good as single-pass on F1
    sp = data["results"]["single_pass"]["metrics"]
    assert full["f1_score"] > sp["f1_score"]
    assert full["hallucination_rate"] < sp["hallucination_rate"]


def test_plots_present():
    for img in (
        "performance_comparison.png",
        "hallucination_rate.png",
        "latency_vs_performance.png",
        "retrieval_efficiency.png",
    ):
        assert (ROOT / "plots" / img).is_file(), f"missing plot {img}"
