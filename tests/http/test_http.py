# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import struct
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
def serve(files, *, mode="ok"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            raw_range = self.headers.get("Range", "")
            requests.append((self.command, self.path, raw_range))
            if self.path not in files:
                self.send_error(403)
                return
            if mode == "retry_once" and len(requests) == 1:
                self.send_error(503)
                return
            if mode == "identity" and self.headers.get("Accept-Encoding") != "identity":
                self.send_error(406)
                return
            if mode == "redirect" and self.path == "/redirect":
                self.send_response(307)
                self.send_header("Location", "/model")
                self.end_headers()
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
            if mode == "disconnect_once" and len(requests) == 1:
                self.close_connection = True
                payload = payload[:2]
            self.wfile.write(payload)

        def do_HEAD(self):
            requests.append((self.command, self.path, self.headers.get("Range")))
            self.send_error(405)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            worker.join()


def read_one(path: str, name: str):
    with SafetensorsStreamer() as streamer:
        streamer.stream_file(path, tensor_names={name})
        return dict(streamer.get_tensors())


def assert_ranges(requests, path, data):
    header_end = 8 + struct.unpack_from("<Q", data)[0]
    expected_bytes = set(range(header_end)) | set(range(header_end + 8, len(data)))
    requested_bytes = set()
    for method, target, raw_range in requests:
        assert (method, target) == ("GET", path)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", raw_range)
        assert match is not None
        start, end = map(int, match.groups())
        assert 0 <= start <= end < len(data)
        requested_bytes.update(range(start, end + 1))
    assert requested_bytes == expected_bytes


def test_selected_tensor_and_signed_request_target(tmp_path):
    local = tmp_path / "model.safetensors"
    data = model(local, "selected", 20)
    reference = read_one(str(local), "selected")
    with serve({SIGNED_PATH: data}) as (base, requests):
        result = read_one(base + SIGNED_PATH, "selected")
    assert result.keys() == reference.keys() == {"selected"}
    torch.testing.assert_close(result["selected"], reference["selected"], rtol=0, atol=0)
    assert_ranges(requests, SIGNED_PATH, data)


def test_explicit_shards(tmp_path):
    local_a, local_b = tmp_path / "a", tmp_path / "b"
    first = model(local_a, "left", 11)
    second = model(local_b, "right", 22)
    with serve({"/a": first, "/b": second}) as (base, received):
        with SafetensorsStreamer() as streamer:
            streamer.stream_files([base + "/a", base + "/b"], tensor_names={"left", "right"})
            result = dict(streamer.get_tensors())
    assert result.keys() == {"left", "right"}
    for path, key in [(local_a, "left"), (local_b, "right")]:
        torch.testing.assert_close(result[key], read_one(str(path), key)[key], rtol=0, atol=0)
    assert_ranges([r for r in received if r[1] == "/a"], "/a", first)
    assert_ranges([r for r in received if r[1] == "/b"], "/b", second)


def test_requests_identity_encoding(tmp_path):
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode="identity") as (base, requests):
        result = read_one(base + SIGNED_PATH, "x")
    torch.testing.assert_close(result["x"], torch.tensor([[1, 2]], dtype=torch.int32))
    assert_ranges(requests, SIGNED_PATH, data)


def test_redirect_is_not_followed(tmp_path):
    data = model(tmp_path / "model", "x", 1)
    with serve({"/redirect": data, "/model": data}, mode="redirect") as (base, requests):
        with pytest.raises(ValueError, match="File access error"):
            read_one(base + "/redirect", "x")
    assert requests
    assert all(path == "/redirect" for _, path, _ in requests)


@pytest.mark.parametrize("mode", [200, "missing", "bounds", "short", "long", 401, 403, 416])
def test_failed_range_is_reported(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("RUNAI_STREAMER_S3_TIMEOUT", "5")
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode=mode) as (base, requests):
        with pytest.raises(ValueError) as error:
            read_one(base + SIGNED_PATH, "x")
    assert requests == [("GET", SIGNED_PATH, "bytes=0-7")]
    if mode in (200, "missing", "bounds"):
        assert "expected 206" in str(error.value)
    if mode in ("short", "long"):
        assert "unexpected number of bytes" in str(error.value)


def test_failed_tensor_read_surfaces_in_get_tensors(tmp_path):
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode="deny_tensor") as (base, requests):
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(base + SIGNED_PATH, tensor_names={"x"})
            with pytest.raises(ValueError, match="File access error"):
                next(iter(streamer.get_tensors()))
    assert_ranges(requests, SIGNED_PATH, data)


def test_retries_transient_range_with_application_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNAI_STREAMER_S3_TIMEOUT", "5")
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode="retry_once") as (base, requests):
        result = read_one(base + SIGNED_PATH, "x")
    torch.testing.assert_close(result["x"], torch.tensor([[1, 2]], dtype=torch.int32))
    assert requests[:2] == [("GET", SIGNED_PATH, "bytes=0-7")] * 2
    assert_ranges(requests[1:], SIGNED_PATH, data)


def test_retries_range_after_premature_eof(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNAI_STREAMER_S3_TIMEOUT", "5")
    data = model(tmp_path / "model", "x", 1)
    with serve({SIGNED_PATH: data}, mode="disconnect_once") as (base, requests):
        result = read_one(base + SIGNED_PATH, "x")
    torch.testing.assert_close(result["x"], torch.tensor([[1, 2]], dtype=torch.int32))
    assert requests[:2] == [("GET", SIGNED_PATH, "bytes=0-7")] * 2
    assert_ranges(requests[1:], SIGNED_PATH, data)


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
