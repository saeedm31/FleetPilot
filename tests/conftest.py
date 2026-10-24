import pytest

from fleetpilot import config as C
from fleetpilot.pipeline import build


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """A small, fast build (400 customers) shared by all tests."""
    data = tmp_path_factory.mktemp("data")
    s = C.Settings(data_dir=data, mlflow_uri=None)
    metrics = build(s, n=400)
    return s, metrics
