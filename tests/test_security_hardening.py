"""Tests for the cleaners security hardening (safe argv, resource caps, safe writes)."""

from __future__ import annotations

import io
import os
import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common
import container_meta
from common import (
    backup_path,
    read_text_input,
    safe_arg,
    safe_write_bytes,
    safe_write_text,
)
from container_meta import (
    MAX_ZIP_DECOMPRESSED_BYTES,
    ZipBudgetExceeded,
    _check_zip_budget,
    _read_zip_member,
    inspect_docx,
)


def test_safe_arg_prefixes_leading_dash():
    assert safe_arg("-@evil") == "./-@evil"
    assert safe_arg("--argfile") == "./--argfile"


def test_safe_arg_leaves_normal_paths_alone():
    assert safe_arg("photo.png") == "photo.png"
    assert safe_arg("dir/file.svg") == "dir/file.svg"
    assert safe_arg("/abs/path.pdf") == "/abs/path.pdf"
    assert safe_arg(".") == "."


def test_zip_budget_rejects_oversized_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        info = zf.infolist()[0]
    info.file_size = MAX_ZIP_DECOMPRESSED_BYTES + 1
    raised = False
    try:
        _check_zip_budget(info, [0])
    except ZipBudgetExceeded:
        raised = True
    assert raised


def _patch_zip_declared_size(data: bytes, new_size: int) -> bytes:
    """Rewrite the uncompressed-size field of every central-directory entry.

    Produces the crafted-archive shape zip-bomb guards must defend
    against: the central directory (which backs ZipInfo.file_size) claims
    ``new_size`` decompressed bytes while the real deflate payload and its
    CRC stay intact. The local headers are left untouched.
    """
    out = bytearray(data)
    pos = 0
    while True:
        pos = out.find(b"PK\x01\x02", pos)
        if pos == -1:
            break
        out[pos + 24 : pos + 28] = struct.pack("<I", new_size)
        pos += 4
    return bytes(out)


def test_zip_budget_charges_real_bytes_not_declared(monkeypatch):
    # The budget must be charged on bytes actually produced, not on the
    # central directory claim (ZipInfo.file_size is attacker-controlled).
    # Here the archive declares 150 KiB for a member that really carries
    # 128 KiB; the accounting must follow reality.
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1 << 20)
    payload = b"y" * (1 << 17)  # 128 KiB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("word/document.xml", payload)
    crafted = _patch_zip_declared_size(buf.getvalue(), (1 << 17) + (1 << 16))
    with zipfile.ZipFile(io.BytesIO(crafted)) as zf:
        info = zf.infolist()[0]
        assert info.file_size == (1 << 17) + (1 << 16)  # declared claim
        budget = [0]
        out = _read_zip_member(zf, info, budget)
    assert out == payload
    assert budget[0] == len(payload)  # real bytes, not the declared claim


def test_zip_budget_accumulates_real_bytes_across_members(monkeypatch):
    # The cap is cumulative across members: several members under the cap
    # individually can still exhaust it together.
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1 << 20)
    payload = b"a" * (1 << 19)  # 512 KiB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.xml", payload)
        zf.writestr("b.xml", payload)
        zf.writestr("c.xml", payload)  # 3 x 512 KiB > 1 MiB cap
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        budget = [0]
        infos = zf.infolist()
        _read_zip_member(zf, infos[0], budget)
        _read_zip_member(zf, infos[1], budget)
        with pytest.raises(ZipBudgetExceeded):
            _read_zip_member(zf, infos[2], budget)


def test_inspect_docx_with_ai_markers_does_not_crash():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
        zf.writestr(
            "docProps/app.xml",
            "<Properties><Application>Claude AI Writer</Application></Properties>",
        )
        zf.writestr(
            "docProps/core.xml",
            "<cp:coreProperties><dc:creator>Anthropic</dc:creator></cp:coreProperties>",
        )
    has_c2pa, has_ai, findings, _ = inspect_docx(buf.getvalue())
    assert has_ai
    assert findings
    assert not has_c2pa or has_ai


# ---------------------------------------------------------------------------
# Safe (atomic, symlink-safe) writes
# ---------------------------------------------------------------------------


def _make_symlink(dest: Path, target: Path) -> None:
    """Create a symlink, skipping where the platform denies the privilege."""
    try:
        dest.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_safe_write_refuses_symlink_destination(tmp_path: Path):
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS DATA")
    dest = tmp_path / "out.txt"
    _make_symlink(dest, victim)
    with pytest.raises(OSError):
        safe_write_text(dest, "cleaned content")
    # The victim must be untouched and no temp litter may remain.
    assert victim.read_text() == "PRECIOUS DATA"
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_write_atomically_replaces_existing_file(tmp_path: Path):
    dest = tmp_path / "out.txt"
    safe_write_text(dest, "first")
    safe_write_text(dest, "second")
    assert dest.read_text() == "second"
    # No stray temp files after a successful write.
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_write_bytes_creates_parent_dirs(tmp_path: Path):
    dest = tmp_path / "a" / "b" / "out.bin"
    safe_write_bytes(dest, b"\x00\x01")
    assert dest.read_bytes() == b"\x00\x01"


def test_safe_write_bytes_without_fchmod(tmp_path: Path, monkeypatch):
    # os.fchmod is POSIX-only; on Windows the write must still go through.
    monkeypatch.delattr(os, "fchmod", raising=False)
    dest = tmp_path / "out.bin"
    safe_write_bytes(dest, b"payload")
    assert dest.read_bytes() == b"payload"
    assert not list(tmp_path.glob("*.tmp"))


def test_backup_path_creates_bak_copy(tmp_path: Path):
    src = tmp_path / "doc.md"
    src.write_text("body")
    bak = backup_path(src)
    assert bak.name == "doc.md.bak"
    assert bak.read_text() == "body"
    assert src.read_text() == "body"


def test_backup_path_refuses_symlinked_bak(tmp_path: Path):
    src = tmp_path / "doc.md"
    src.write_text("body")
    bak = tmp_path / "doc.md.bak"
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS")
    _make_symlink(bak, victim)
    with pytest.raises(SystemExit):
        backup_path(src)
    assert victim.read_text() == "PRECIOUS"


# ---------------------------------------------------------------------------
# Input size caps (stdin + file)
# ---------------------------------------------------------------------------


def test_read_text_input_refuses_oversized_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "MAX_INPUT_BYTES", 8)
    big = tmp_path / "big.txt"
    big.write_text("x" * 64)
    with pytest.raises(SystemExit):
        read_text_input(str(big))


def test_read_stdin_capped(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "MAX_STDIN_BYTES", 16)
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 64))
    with pytest.raises(SystemExit):
        read_text_input(None)


def test_read_stdin_under_cap_ok(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "MAX_STDIN_BYTES", 1024)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello stdin"))
    assert read_text_input(None) == "hello stdin"


def test_reconfigure_stream_writes_utf8():
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    common._reconfigure_stream(stream, "backslashreplace")
    stream.write("\u200b")
    stream.flush()
    assert buf.getvalue() == "\u200b".encode("utf-8")
