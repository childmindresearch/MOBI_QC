"""Fixtures used by pytest."""

import pathlib

import pytest

from MOBI_QC.core.models import DataStream
from MOBI_QC.io.readers import readers


@pytest.fixture
def sample_xdf_file() -> pathlib.Path:
    """Test data for XDF files."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / "sub-P001_ses-S001_task-Default_run-001_mobi.xdf"
    )


@pytest.fixture
def sample_datastream_obj(sample_xdf_file: pathlib.Path) -> DataStream.DataStream:
    """Create test DataStream object from sample XDF file."""
    stream = readers.read_xdf(xdf_path=sample_xdf_file, stream_names=["GazeStream"])[0]
    return DataStream.DataStream(stream=stream)
