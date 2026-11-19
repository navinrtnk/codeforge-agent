"""Smoke-test the shipped API image under Compose-like filesystem restrictions."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pytest


@pytest.mark.skipif(os.getenv("CODEFORGE_DOCKER_TESTS") != "1", reason="Docker opt-in required")
def test_application_container_health_errors_and_usage() -> None:
    name = f"codeforge-api-test-{uuid.uuid4().hex}"
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=64m",
                "--tmpfs",
                "/data:rw,nosuid,nodev,size=32m,mode=1777",
                "-p",
                "127.0.0.1::8000",
                "codeforge-agent:local",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        port = subprocess.run(
            ["docker", "port", name, "8000"], capture_output=True, text=True, check=True
        ).stdout.strip()
        base = f"http://{port}"
        for _ in range(100):
            try:
                with urllib.request.urlopen(base + "/health", timeout=1) as response:
                    assert json.load(response) == {"status": "ok"}
                    assert response.headers["x-request-id"]
                break
            except urllib.error.URLError, TimeoutError, ConnectionError:
                time.sleep(0.1)
        else:
            raise AssertionError("Container did not become healthy")
        with urllib.request.urlopen(base + "/usage", timeout=2) as response:
            assert json.load(response) == {"models": []}
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/missing", timeout=2)
        assert error.value.code == 404
        assert json.load(error.value)["code"] == "http_404"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False, timeout=30)
