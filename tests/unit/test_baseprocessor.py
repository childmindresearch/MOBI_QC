"""Unit tests for BaseProcessor class."""

import pathlib

import polars as pl

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
    requested_names = ["EventMarkerStream", "GazeStream"]
    processor = BaseProcessor.BaseProcessor(
        xdf_path=sample_xdf_file, stream_names=requested_names
    )

    assert processor.subject_id == "sub-P001"
    assert isinstance(processor.raw_data, list)
    assert processor.stream_names == requested_names


def test_baseprocessor_format_data(sample_xdf_file: str | pathlib.Path) -> None:
    """Test BaseProcessor format_data method."""
    processor = BaseProcessor.BaseProcessor(xdf_path=sample_xdf_file)
    processor.format_data()

    for stream in processor.raw_data:
        stream_type = stream["info"]["type"][0]
        assert hasattr(processor, stream_type)
        data_stream = getattr(processor, stream_type)
        assert data_stream.data_modality == stream_type
        assert isinstance(data_stream.data, pl.DataFrame)
        assert (
            data_stream.data.get_column("time_stamp").to_numpy()
            == stream["time_stamps"]
        ).all()
