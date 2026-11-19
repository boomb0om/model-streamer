# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import ssl
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from runai_model_streamer import FileStreamer, SafetensorsStreamer, list_safetensors
from runai_model_streamer.file_streamer.file_streamer import RunaiStreamerInvalidInputException

# The server matches the raw request target, including order, duplicate keys, escaping and dot segments.
SIGNED_PATH = "/gateway/../model%2fpart?tag=b&sig=test%2Bsignature&tag=a"


def model(path: Path, selected: str, value: int) -> bytes:
    header = json.dumps({
        "skipped": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]},
        selected: {"dtype": "I32", "shape": [1, 2], "data_offsets": [8, 16]},
    }).encode()
    header += b" " * (-len(header) % 8)
    data = struct.pack("<Q", len(header)) + header + struct.pack("<4i", -10, -11, value, value + 1)
    path.write_bytes(data)
    return data


@contextmanager
def serve(files, *, mode="ok", redirect=None, tls=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            raw_range = self.headers.get("Range", "")
            requests.append((self.command, self.path, raw_range))
            if self.path not in files:
                self.send_error(403)
                return
            if redirect:
                self.send_response(307)
                self.send_header("Location", redirect)
                self.end_headers()
                self.wfile.write(b"redirect body")
                return
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", raw_range)
            if not match:
                self.send_error(400)
                return
            start, end = map(int, match.groups())
            data = files[self.path]
            if mode == "deny_tensor" and start >= 8 + struct.unpack_from("<Q", data)[0]:
                self.send_error(403)
                return
            if mode in (401, 403, 416):
                self.send_error(mode)
                return
            self.send_response(200 if mode == 200 else 206)
            if mode != "missing":
                first = start + 1 if mode == "bounds" else start
                self.send_header("Content-Range", f"bytes {first}-{end}/{len(data)}")
            payload = data if mode == 200 else data[start:end + 1]
            if mode == "short":
                payload = payload[:-1]
            if mode == "long":
                payload += b"x"
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_HEAD(self):
            requests.append((self.command, self.path, self.headers.get("Range")))
            self.send_error(405)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        if tls:
            server.socket = tls.wrap_socket(server.socket, server_side=True)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            yield f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            worker.join()


def read_one(path: str, name: str):
    with SafetensorsStreamer() as streamer:
        streamer.stream_file(path, tensor_names={name})
        return dict(streamer.get_tensors())


def assert_ranges(requests, path, data):
    header_end = 8 + struct.unpack_from("<Q", data)[0]
    assert sorted(requests) == sorted([
        ("GET", path, "bytes=0-7"),
        ("GET", path, f"bytes=8-{header_end - 1}"),
        ("GET", path, f"bytes={header_end + 8}-{header_end + 15}"),
    ])


def test_selected_tensor_and_signed_request_target(tmp_path):
    local = tmp_path / "model.safetensors"
    data = model(local, "selected", 20)
    reference = read_one(str(local), "selected")
    with serve({SIGNED_PATH: data}) as (base, requests):
        result = read_one(base + SIGNED_PATH, "selected")
    assert result.keys() == reference.keys() == {"selected"}
    torch.testing.assert_close(result["selected"], reference["selected"], rtol=0, atol=0)
    assert_ranges(requests, SIGNED_PATH, data)


@pytest.fixture
def certificate(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1",
        "-keyout", str(key), "-out", str(cert),
    ], capture_output=True, check=True)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    return cert, tls


def test_explicit_mixed_scheme_shards_and_https_redirect(tmp_path, certificate, monkeypatch):
    cert, tls = certificate
    local_a, local_b = tmp_path / "a", tmp_path / "b"
    first = model(local_a, "left", 11)
    second = model(local_b, "right", 22)
    monkeypatch.setenv("CURL_CA_BUNDLE", str(cert))
    with serve({"/a": first, "/b": second}, tls=tls) as (secure, received):
        with serve({"/redirect": first}, redirect=secure + "/a") as (plain, redirected):
            with SafetensorsStreamer() as streamer:
                streamer.stream_files([plain + "/redirect", secure + "/b"], tensor_names={"left", "right"})
                result = dict(streamer.get_tensors())
    assert result.keys() == {"left", "right"}
    for path, key in [(local_a, "left"), (local_b, "right")]:
        torch.testing.assert_close(result[key], read_one(str(path), key)[key], rtol=0, atol=0)
    assert_ranges(redirected, "/redirect", first)
    assert_ranges([r for r in received if r[1] == "/a"], "/a", first)
    assert_ranges([r for r in received if r[1] == "/b"], "/b", second)


def test_untrusted_redirect_fails_before_get(tmp_path, certificate, monkeypatch):
    _, tls = certificate
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    data = model(tmp_path / "model", "x", 1)
    with serve({"/model": data}, tls=tls) as (secure, received):
        with serve({"/redirect": data}, redirect=secure + "/model") as (plain, redirected):
            with pytest.raises(ValueError, match="File access error"):
                read_one(plain + "/redirect", "x")
    assert redirected == [("GET", "/redirect", "bytes=0-7")]
    assert received == []


@pytest.mark.parametrize("mode", [200, "missing", "bounds", "short", "long", 401, 403, 416])
def test_failed_range_is_reported(tmp_path, mode):
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode=mode) as (base, requests):
        with pytest.raises(ValueError) as error:
            read_one(base + SIGNED_PATH, "x")
    assert requests == [("GET", SIGNED_PATH, "bytes=0-7")]
    if mode in (200, "missing", "bounds"):
        assert "expected 206" in str(error.value)
    assert "signature" not in str(error.value)


def test_failed_tensor_read_surfaces_in_get_tensors(tmp_path):
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode="deny_tensor") as (base, requests):
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(base + SIGNED_PATH, tensor_names={"x"})
            with pytest.raises(ValueError, match="File access error"):
                next(iter(streamer.get_tensors()))
    assert_ranges(requests, SIGNED_PATH, data)


def test_signed_query_absent_from_python_and_native_diagnostics():
    # Logging configuration is read at library load. Start a fresh interpreter with both loggers on
    # and a valid 206 carrying a corrupt safetensors length so the Python exception names the path.
    with serve({SIGNED_PATH: b"\xff" * 8}) as (base, requests):
        result = subprocess.run([sys.executable, "-c", """
import logging
import sys
logging.addLevelName(5, "SPAM")  # Enable the native logger's most verbose level in Python too.
from runai_model_streamer import SafetensorsStreamer
logging.basicConfig(level=logging.DEBUG)
with SafetensorsStreamer() as streamer:
    streamer.stream_file(sys.argv[1], tensor_names={"x"})
""", base + SIGNED_PATH], env=dict(os.environ, RUNAI_STREAMER_LOG_LEVEL="SPAM",
                                   RUNAI_STREAMER_LOG_TO_STDERR="1"), capture_output=True, text=True)
    assert requests == [("GET", SIGNED_PATH, "bytes=0-7")]
    assert result.returncode != 0
    assert "exceeds limit" in result.stderr
    assert "signature" not in result.stdout + result.stderr


def test_directory_listing_explains_how_to_supply_shards():
    with pytest.raises(NotImplementedError, match="explicit file URLs"):
        list_safetensors("https://example.invalid/models/")
    with pytest.raises(ValueError, match="explicit file URLs"):
        FileStreamer().list_files("https://example.invalid/models/")


@pytest.mark.parametrize("other", ["/local/model", "s3://bucket/model"])
def test_mixed_sources_rejected_before_read(other):
    with SafetensorsStreamer() as streamer:
        with pytest.raises(RunaiStreamerInvalidInputException, match="multiple source types"):
            streamer.stream_files(["http://127.0.0.1/model", other], tensor_names={"x"})
