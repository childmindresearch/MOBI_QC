"""Unit tests for the DataStream class."""

import numpy as np
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
    onset_timestamp = np.random.uniform(
        low=1.0, high=sample_datastream_obj.data.item(-1, "time_stamp")
    )
    offset_timestamp = np.random.uniform(low=0.0, high=onset_timestamp)
    with pytest.raises(
        ValueError, match="Offset timestamp must be greater than onset timestamp."
    ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)


def test_raises_value_error_if_offset_onset_equal(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that ValueError is raised when offset and onset timestamps are equal."""
    onset_timestamp = offset_timestamp = np.random.uniform(
        low=0, high=sample_datastream_obj.data.item(-1, "time_stamp")
    )
    with pytest.raises(
        ValueError, match="Offset timestamp must be greater than onset timestamp."
    ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)


def test_raises_value_error_if_timestamps_negative(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that ValueError is raised when onset or offset timestamps are negative."""
    onset_timestamp = np.random.uniform(low = -10000, high=-2.0)
    offset_timestamp = np.random.uniform(low=onset_timestamp, high=-1.0)
    with pytest.raises(
        ValueError, match="Onset and offset timestamps must be positive values."
    ):
        sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)

def test_fs_zero_if_df_empty(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test that empty DF yields sampling rate of 0."""
    onset_timestamp = np.random.uniform(
        low = sample_datastream_obj.data.item(-1, "time_stamp"), high = 10000
        )
    offset_timestamp = np.random.uniform(low = onset_timestamp, high = 10000)
    sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)
    assert sample_datastream_obj.effective_srate == 0

def test_datastream_filter_time_range(
    sample_datastream_obj: DataStream.DataStream,
) -> None:
    """Test DataStream filter_time_range method."""
    onset_index = np.random.randint(low=0, high=len(sample_datastream_obj.data) - 3)
    offset_index = np.random.randint(
        low=onset_index + 1, high=len(sample_datastream_obj.data) - 1
    )
    onset_timestamp = sample_datastream_obj.data.item(onset_index, "time_stamp")
    offset_timestamp = sample_datastream_obj.data.item(offset_index, "time_stamp")
    expected_duration = offset_timestamp - onset_timestamp

    sample_datastream_obj.filter_time_range(onset_timestamp, offset_timestamp)
    new_duration = sample_datastream_obj.data.item(
        -1, "time_stamp"
    ) - sample_datastream_obj.data.item(0, "time_stamp")
    expected_fs = 1 / (
        sample_datastream_obj.data.select(pl.col("time_stamp").diff()).mean().item()
    )

    assert new_duration == expected_duration
    assert sample_datastream_obj.effective_srate == expected_fs
