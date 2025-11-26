"""Fixtures used by pytest."""

import pathlib

import pytest


@pytest.fixture
def sample_xdf_file() -> pathlib.Path:
    """Test data for XDF files."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / "sub-P001_ses-S001_task-Default_run-001_mobi.xdf"
    )
