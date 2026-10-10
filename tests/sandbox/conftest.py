"""Shared fixtures for the docker-marked sandbox tests."""

from __future__ import annotations

import shutil

import pytest

from ds_research_agent.config import SandboxSettings
from ds_research_agent.sandbox import SandboxRunner


@pytest.fixture(scope="session")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    if shutil.which("docker") is None:
        pytest.fail("docker CLI not found; the docker-marked tests need Docker")
    settings = SandboxSettings(
        docker="docker",
        image="dsra-sandbox:test",
        work_root=tmp_path_factory.mktemp("sandbox"),
        wall_timeout_s=8,
        host_grace_s=30,
        cpu_time_s=3,
        max_file_bytes=1_000_000,
        memory_mb=512,
        cpus=1,
        pids_limit=64,
        tmp_mb=16,
        max_output_chars=2_000,
    )
    r = SandboxRunner(settings)
    r.build_image()
    return r
