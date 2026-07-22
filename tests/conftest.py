import pytest

from src.config.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_persist_dir=tmp_path)
