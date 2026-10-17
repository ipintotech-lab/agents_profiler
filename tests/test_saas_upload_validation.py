import gzip
import io
import json
import socket
import struct
import threading
import zipfile

import pytest

from saas.scanner import ClamdScanner, NoScanner
from saas.storage import LocalObjectStorage, object_key
from saas.validation import MAX_FILENAME_CHARS, Rejected, detect, display_filename

T = "a" * 32
O = "b" * 32


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\file.json", "file.json"),
        ("/abs/path/run.jsonl", "run.jsonl"),
        ("..", "upload"),
        ("", "upload"),
        (None, "upload"),
        ("evil" + chr(0x202E) + "gnp.json", "evilgnp.json"),
        ("tab\tnew\nline.csv", "tabnewline.csv"),
        ("café.json", "café.json"),
    ],
)
def test_display_filename_is_metadata_safe(raw, expected):
    assert display_filename(raw) == expected


def test_long_filenames_are_capped_and_keep_their_extension():
    name = display_filename("x" * 5000 + ".jsonl")
    assert len(name) <= MAX_FILENAME_CHARS and name.endswith(".jsonl")


def _zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.json", "{}")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("data", "name", "reason"),
    [
        (b"", "a.json", "empty"),
        (b"   \n", "a.json", "no content"),
        (_zip(), "a.json", "ZIP"),
        (gzip.compress(b'{"a": 1}'), "a.jsonl", "gzip"),
        (b"\x7fELF\x02\x01", "a.json", "executable"),
        (b"MZ\x90\x00", "run.csv", "executable"),
        (b"%PDF-1.7", "a.json", "PDF"),
        (b'{"a": "\x00"}', "a.json", "binary"),
        (b"\xff\xfe{\x00}\x00", "a.json", "binary"),
        (b'{"a": "\xc3\x28"}', "a.json", "UTF-8"),
        (b'{"a": 1}', "a.exe", "extension"),
        (b'{"a": 1}', "a.json.exe", "extension"),
        (b"#!/bin/sh\nrm -rf /\n", "a.json", "not valid"),
        (b"<html><script>x</script></html>", "a.json", "not valid"),
        (b"a,b\n1,2\n", "a.json", "not valid"),
        (b'{"a": 1}', "a.csv", "named as CSV"),
        (b"[" * 200 + b"]" * 200, "a.json", "nested"),
    ],
)
def test_content_decides_and_bad_files_are_rejected(data, name, reason):
    with pytest.raises(Rejected, match=reason):
        detect(data, name)


def test_rejection_messages_never_quote_the_content():
    secret = b"sk-live-not-a-real-key-but-pretend"
    with pytest.raises(Rejected) as exc:
        detect(secret, "a.json")
    assert secret.decode() not in str(exc.value)


@pytest.mark.parametrize(
    ("data", "name", "kind"),
    [
        (json.dumps([{"a": 1}]).encode(), "run.json", "json"),
        (b'{"a": 1}\n{"b": 2}\n', "run.jsonl", "jsonl"),
        (b'{"a": 1}\n{"b": 2}\n', "run.json", "jsonl"),
        (b'\xef\xbb\xbf{"a": 1}', "run.json", "json"),
        (b"step,tokens\nread,10\nwrite,12\n", "usage.CSV", "csv"),
    ],
)
def test_supported_files_are_detected_from_content(data, name, kind):
    assert detect(data, name).type == kind


def test_storage_keys_are_server_shaped_and_confined(tmp_path):
    store = LocalObjectStorage(tmp_path)
    key = object_key(T, "quarantine", O)
    store.put(key, b"data")
    assert store.get(key) == b"data"
    assert (tmp_path / key).stat().st_mode & 0o777 == 0o600
    dst = object_key(T, "uploads", O)
    store.move(key, dst)
    assert not store.exists(key) and store.get(dst) == b"data"
    store.delete(dst)
    assert not store.exists(dst)
    for bad in ("../x", f"tenants/{T}/uploads/../../../etc/passwd", f"tenants/{T}/other/{O}", "/etc/passwd"):
        with pytest.raises(ValueError):
            store.get(bad)
    with pytest.raises(ValueError):
        object_key("../" + T, "uploads", O)


def _fake_clamd(reply: bytes):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    received = []

    def serve():
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while not buf.endswith(struct.pack("!L", 0)):
                buf += conn.recv(65536)
            received.append(buf)
            conn.sendall(reply)
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], received


def test_clamd_scanner_reports_clean_and_infected():
    port, received = _fake_clamd(b"stream: OK\0")
    assert ClamdScanner("127.0.0.1", port).scan(b"hello").clean
    assert received[0].startswith(b"zINSTREAM\0")
    port, _ = _fake_clamd(b"stream: Eicar-Test-Signature FOUND\0")
    result = ClamdScanner("127.0.0.1", port).scan(b"x")
    assert not result.clean and result.signature == "Eicar-Test-Signature"


def test_no_scanner_marks_files_unscanned():
    assert NoScanner().scan(b"x").scanned is False
