import os
from pathlib import Path

import pytest

from ds_research_agent.config import Settings, load_settings

DEFAULT = Path(__file__).parents[2] / "config" / "local.yaml"


@pytest.fixture(scope="session")
def settings() -> Settings:
    path = Path(os.environ.get("DSRA_CONFIG", DEFAULT))
    if not path.exists():
        pytest.skip(f"no local config at {path}")
    return load_settings(path)
