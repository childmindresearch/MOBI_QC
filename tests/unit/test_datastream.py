"""Unit tests for the DataStream class."""

import math

import polars as pl
import pytest

from MOBI_QC.core.models import DataStream


def test_datastream_initialization(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test DataStream initialization and attribute assignments."""
    assert isinstance(sample_datastream_obj.stream_name, str)
    assert isinstance(sample_datastream_obj.data, pl.DataFrame)
    assert isinstance(sample_datastream_obj.variables, list)
    assert isinstance(sample_datastream_obj.data_modality, str)
    assert isinstance(sample_datastream_obj.channel_count, int)
    assert isinstance(sample_datastream_obj.nominal_srate, float)
    assert isinstance(sample_datastream_obj.source_id, str)
    assert isinstance(sample_datastream_obj.uid, str)
    assert isinstance(sample_datastream_obj.effective_srate, float)
    assert isinstance(sample_datastream_obj.desc, dict)
    assert hasattr(sample_datastream_obj, "qc_metrics")
    assert isinstance(sample_datastream_obj.qc_metrics, dict)
    assert hasattr(sample_datastream_obj, "error")
    assert not sample_datastream_obj.error


def test_raises_value_error_if_offset_less(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that ValueError is raised when offset timestamp is less than onset."""
    onset_timestamp = 100.0
    offset_timestamp = 10.0
    with pytest.raises(
        ValueError, match="Offset timestamp must be greater than onset timestamp."
    ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)


def test_raises_value_error_if_offset_onset_equal(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that ValueError is raised when offset and onset timestamps are equal."""
    timestamp = 100.0
    with pytest.raises(
        ValueError, match="Offset timestamp must be greater than onset timestamp."
    ):
        sample_datastream_obj.filter_time_range(timestamp, timestamp)


def test_raises_value_error_if_timestamps_negative(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that ValueError is raised when onset or offset timestamps are negative."""
    onset_timestamp = -100.0
    offset_timestamp = -10.0
    with pytest.raises(
        ValueError, match="Onset and offset timestamps must be positive values."
    ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)


def test_fs_zero_if_df_empty(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that an empty DF yields sampling rate of 0."""
    onset_timestamp = sample_datastream_obj.data.item(-1, "time_stamp") + 10
    offset_timestamp = onset_timestamp + 100
    sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)
    assert sample_datastream_obj.effective_srate == 0


def test_datastream_filter_time_range(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test DataStream filter_time_range method."""
    onset_index = 10
    offset_index = 100
    onset_timestamp = sample_datastream_obj.data.item(onset_index, "time_stamp")
    offset_timestamp = sample_datastream_obj.data.item(offset_index, "time_stamp")
    expected_duration = offset_timestamp - onset_timestamp
    expected_fs = 50.4754  # known value from sample data

    sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)
    new_duration = sample_datastream_obj.data.item(
        -1, "time_stamp"
    ) - sample_datastream_obj.data.item(0, "time_stamp")

    assert new_duration == expected_duration
    assert math.isclose(sample_datastream_obj.effective_srate, expected_fs, \
        rel_tol=10e-7)