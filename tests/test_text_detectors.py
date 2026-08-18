"""Tests for text_detectors.py (vendor/research text-watermark detectors)."""

from __future__ import annotations

import http.client
import io
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import text_detectors


class _FakeResp:
    def __init__(self, data: bytes | dict):
        self._data = json.dumps(data).encode("utf-8") if isinstance(data, dict) else data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


def _http_error(code: int) -> urllib.error.HTTPError:
    hdrs = http.client.HTTPMessage()
    return urllib.error.HTTPError(
        "http://generativelanguage.invalid", code, "err", hdrs, io.BytesIO(b'{"error": "denied"}')
    )


def _gemini_success(verdict: str | None = None, score: float | None = None) -> dict:
    candidate: dict = {}
    if verdict is not None:
        candidate["content"] = {"parts": [{"text": verdict}]}
    if score is not None:
        candidate["attributionMetadata"] = {"syntheticText": {"score": score}}
    return {"candidates": [candidate]}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "WATERMARKS_GEMINI_API_KEY",
        "WATERMARKS_GEMINI_MODEL",
        "WATERMARKS_GEMINI_MAX_CHARS",
        "WATERMARKS_MARKLLM_SCHEME",
        "MARKLLM_DIR",
        "WATERMARKS_MARKLLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


# --- Gemini ----------------------------------------------------------------


def test_gemini_unconfigured():
    report = text_detectors.GeminiSynthIDTextDetector().detect("hello")
    assert report["available"] is False
    assert "WATERMARKS_GEMINI_API_KEY" in report["error"]
    assert text_detectors.GeminiSynthIDTextDetector().available() is False


def test_gemini_verdict_watermarked(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(_gemini_success(verdict="Likely AI-generated")),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["verdict"] == "Likely AI-generated"


def test_gemini_verdict_unlikely(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(_gemini_success(verdict="Unlikely AI-generated")),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["is_watermarked"] is False


def test_gemini_numeric_score(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(_gemini_success(score=0.87)),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["is_watermarked"] is True
    assert report["score"] == 0.87


def test_gemini_http_error(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(401)),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is False
    assert "HTTP 401" in report["error"]


def test_gemini_retries_once_on_429(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429)
        return _FakeResp(_gemini_success(verdict="Likely AI-generated"))

    monkeypatch.setattr(text_detectors.urllib.request, "urlopen", flaky)
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert calls["n"] == 2
    assert report["is_watermarked"] is True


def test_gemini_oversize_skips(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_GEMINI_MAX_CHARS", "10")
    report = text_detectors.GeminiSynthIDTextDetector().detect("x" * 100)
    assert report["available"] is True
    assert report["skipped"] is True
    assert report["is_watermarked"] is None


def test_gemini_malformed_max_chars_env_does_not_crash(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_GEMINI_MAX_CHARS", "not-a-number")
    monkeypatch.setattr(
        text_detectors.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(_gemini_success(verdict="Likely AI-generated")),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("short text")
    assert report["available"] is True
    assert report["is_watermarked"] is True


def test_gemini_no_candidates(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors.urllib.request, "urlopen", lambda *a, **k: _FakeResp({"candidates": []})
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is False
    assert "no candidates" in report["error"]


# --- MarkLLM ---------------------------------------------------------------


def test_markllm_unconfigured():
    assert text_detectors.MarkLLMTextDetector().available() is False
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is False
    assert "MARKLLM_DIR" in report["error"]


def test_markllm_success(monkeypatch):
    monkeypatch.setenv("MARKLLM_DIR", "/fake/MarkLLM")
    payload = {"is_watermarked": True, "score": 4.2, "threshold": 4.0}
    monkeypatch.setattr(
        text_detectors.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=json.dumps(payload)),
    )
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert "research harness" in report["note"]


def test_markllm_unavailable_exit3(monkeypatch):
    monkeypatch.setenv("MARKLLM_DIR", "/fake/MarkLLM")
    monkeypatch.setattr(
        text_detectors.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 3, stdout="", stderr="missing deps"),
    )
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is False
    assert "missing deps" in report["error"]


def test_markllm_scheme_env(monkeypatch):
    monkeypatch.setenv("MARKLLM_DIR", "/fake/MarkLLM")
    monkeypatch.setenv("WATERMARKS_MARKLLM_SCHEME", "synthid")
    seen = {}

    def fake_run(*args, **kwargs):
        seen["argv"] = args[0]
        return subprocess.CompletedProcess(args[0], 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    text_detectors.MarkLLMTextDetector().detect("hello")
    assert "--scheme" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--scheme") + 1] == "synthid"


def test_markllm_prefers_checkout_venv(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    if os.name == "nt":
        venv_python = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = upstream / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    assert det.available() is True
    det.detect("hello")
    assert seen["argv"][0] == str(venv_python)


def test_markllm_falls_back_to_sys_executable(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    det.detect("hello")
    assert seen["argv"][0] == sys.executable


def test_markllm_ctor_overrides_passed_to_adapter(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(
        scheme="synthid",
        upstream_dir=str(upstream),
        model="opt-1.3b",
        timeout=5,
    )
    det.detect("hello")
    argv = seen["argv"]
    assert argv[argv.index("--scheme") + 1] == "synthid"
    assert argv[argv.index("--model") + 1] == "opt-1.3b"
    assert argv[argv.index("--upstream-dir") + 1] == str(upstream.resolve())


def test_markllm_ctor_available_with_override(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    assert det.available() is True


def test_markllm_preexec_default_off(monkeypatch):
    monkeypatch.delenv("WATERMARKS_MARKLLM_RLIMIT_AS", raising=False)
    assert text_detectors._markllm_preexec() is None


def test_markllm_preexec_env(monkeypatch):
    if os.name != "posix":
        pytest.skip("preexec_fn is POSIX-only")
    monkeypatch.setenv("WATERMARKS_MARKLLM_RLIMIT_AS", "0x40000000")
    assert callable(text_detectors._markllm_preexec())


def test_markllm_detect_applies_rlimit(monkeypatch, tmp_path):
    if os.name != "posix":
        pytest.skip("preexec_fn is POSIX-only")
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    monkeypatch.setenv("WATERMARKS_MARKLLM_RLIMIT_AS", "1073741824")
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["preexec_fn"] = kwargs.get("preexec_fn")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"available": true}')

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    assert report["available"] is True
    assert callable(captured["preexec_fn"])


def test_run_all_text_detectors_can_exclude_markllm(monkeypatch):
    monkeypatch.setattr(
        text_detectors,
        "MarkLLMTextDetector",
        lambda: pytest.fail("must not construct MarkLLM when excluded"),
    )
    reports = text_detectors.run_all_text_detectors("hello", include_markllm=False)
    assert len(reports) == 2  # gemini (unconfigured) + claude placeholder
    assert {r["detector"] for r in reports} == {"gemini-synthid-text", "claude-text"}


def test_run_all_text_detectors_injects_markllm_instance(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream), scheme="synthid")
    seen: list = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)
    text_detectors.run_all_text_detectors("hello", markllm=det)
    markllm_cmd = next(c for c in seen if "--scheme" in c)
    assert markllm_cmd[markllm_cmd.index("--scheme") + 1] == "synthid"


# --- Claude placeholder ----------------------------------------------------


def test_claude_placeholder():
    det = text_detectors.ClaudeTextDetector()
    assert det.available() is False
    report = det.detect("hello")
    assert report["available"] is False
    assert "WATERMARKS_CLAUDE_API_KEY" in report["error"]


# --- Registry --------------------------------------------------------------


def test_detector_status_keys():
    status = text_detectors.detector_status()
    assert set(status) == {"gemini-synthid-text", "markllm", "claude-text"}


def test_run_all_text_detectors_length():
    reports = text_detectors.run_all_text_detectors("hello")
    assert len(reports) == 3
    assert all("detector" in r for r in reports)


def test_run_text_detectors_filters_unavailable():
    reports = text_detectors.run_text_detectors("hello")
    assert reports == []
