"""Unit tests for BaseProcessor class."""

import pathlib

from MOBI_QC.core.models import BaseProcessor


def test_baseprocessor_initialization(sample_xdf_file: pathlib.Path) -> None:
    """Test BaseProcessor initialization and attribute assignments."""
    processor = BaseProcessor.BaseProcessor(xdf_path=sample_xdf_file)

    assert processor.subject_id == "sub-P001"
    assert isinstance(processor.raw_data, list)
    assert isinstance(processor.stream_names, list)
    assert len(processor.stream_names) > 0


def test_baseprocessor_with_stream_names(sample_xdf_file: str | pathlib.Path) -> None:
    """Test BaseProcessor initialization with specific stream names."""
    requested_names = ["AudioMarkerStream", "GazeStream"]
    processor = BaseProcessor.BaseProcessor(
        xdf_path=sample_xdf_file, stream_names=requested_names
    )

    assert processor.subject_id == "sub-P001"
    assert isinstance(processor.raw_data, list)
    assert processor.stream_names == requested_names


def test_get_collection_date(sample_xdf_file: str | pathlib.Path) -> None:
    """Test the get_collection_date method of BaseProcessor."""
    expected_date = "2025-12-02 16:24:52"
    processor = BaseProcessor.BaseProcessor(xdf_path=sample_xdf_file)
    collection_date = processor.get_collection_date()

    assert collection_date == expected_date
