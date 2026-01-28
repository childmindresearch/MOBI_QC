"""Unit tests for the DataStream class."""

import pathlib

import polars as pl

from MOBI_QC.core.models import DataStream
from MOBI_QC.io.readers import readers


def test_datastream_initialization(sample_xdf_file: pathlib.Path) -> None:
    """Test DataStream initialization and attribute assignments."""
    stream = readers.read_xdf(xdf_path=sample_xdf_file, stream_names=["GazeStream"])[0]

    ds = DataStream.DataStream(stream=stream)

    assert isinstance(ds.stream_name, str)
    assert isinstance(ds.data, pl.DataFrame)
    assert isinstance(ds.variables, list)
    assert isinstance(ds.data_modality, str)
    assert isinstance(ds.channel_count, int)
    assert isinstance(ds.nominal_srate, float)
    assert isinstance(ds.source_id, str)
    assert isinstance(ds.uid, str)
    assert isinstance(ds.effective_srate, float)
    assert isinstance(ds.desc, dict)
    assert hasattr(ds, "qc_metrics")
    assert isinstance(ds.qc_metrics, dict)
    assert hasattr(ds, "error")
    assert not ds.error


def test_datastream_filter_time_range(sample_xdf_file: pathlib.Path) -> None:
    """Test DataStream filter_time_range method."""
    stream = readers.read_xdf(xdf_path=sample_xdf_file, stream_names=["GazeStream"])[0]

    ds = DataStream.DataStream(stream=stream)
    onset_timestamp = ds.data.item(2, "time_stamp")
    offset_timestamp = ds.data.item(200, "time_stamp")

    expected_duration = offset_timestamp - onset_timestamp
    ds.filter_time_range(onset_timestamp, offset_timestamp)
    duration = ds.data.item(-1, "time_stamp") - ds.data.item(0, "time_stamp")

    expected_fs = 1 / (ds.data.select(pl.col("time_stamp").diff()).mean().item())

    assert duration == expected_duration
    assert ds.effective_srate == expected_fs
