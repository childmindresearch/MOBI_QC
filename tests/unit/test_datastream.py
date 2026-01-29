"""Unit tests for the DataStream class."""

import pathlib

import polars as pl
import pytest
import numpy as np

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

def test_raises_value_error_if_offset_less(sample_datastream_obj: DataStream.DataStream) -> None:
    """Test that ValueError is raised when offset timestamp is less than onset."""

    # rand float btw 1 + n
    # rand float btw 0 + onset
    onset_timestamp = 20
    offset_timestamp = 10
    with pytest.raises(
        ValueError, match = "Offset timestamp must be greater than onset timestamp."
        ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)

def test_raises_value_error_if_offset_onset_equal(
    sample_datastream_obj: DataStream.DataStream
    ) -> None:
    """Test that ValueError is raised when offset and onset timestamps are equal."""
    # rand float, they both equal
    onset_timestamp = 20
    offset_timestamp = 20
    with pytest.raises(
        ValueError, match = "Offset timestamp must be greater than onset timestamp."
        ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)

def test_raises_value_error_if_timestamps_negative(
    sample_datastream_obj: DataStream.DataStream
    ) -> None:
    """Test that ValueError is raised when onset or offset timestamps are negative."""
    # rand negative float for both 
    onset_timestamp = -20.0
    offset_timestamp = -10.0
    with pytest.raises(
        ValueError, match = "Onset and offset timestamps must be positive values."
        ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)


def test_datastream_filter_time_range(
    sample_datastream_obj: DataStream.DataStream
    ) -> None:
    """Test DataStream filter_time_range method."""
    onset_index = np.random.randint(low = 0, high = len(sample_datastream_obj.data)-3)
    offset_index = np.random.randint(low = onset_index+1, \
        high = len(sample_datastream_obj.data)-1)
    onset_timestamp = sample_datastream_obj.data.item(onset_index, "time_stamp")
    offset_timestamp = sample_datastream_obj.data.item(offset_index, "time_stamp")
    expected_duration = offset_timestamp - onset_timestamp

    sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)
    new_duration = sample_datastream_obj.data.item(-1, "time_stamp") \
         - sample_datastream_obj.data.item(0, "time_stamp")
    expected_fs = 1 / \
        (sample_datastream_obj.data.select(pl.col("time_stamp").diff()).mean().item())

    assert new_duration == expected_duration
    assert sample_datastream_obj.effective_srate == expected_fs