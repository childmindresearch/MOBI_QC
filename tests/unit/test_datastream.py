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
    assert math.isclose(
        sample_datastream_obj.effective_srate, expected_fs, rel_tol=10e-7
    )


# Tests for calculate_dropped_samples method


def create_test_datastream(
    sample_rate: float, timestamps: list[float]
) -> DataStream.DataStream:
    """Helper function to create a test DataStream with given timestamps."""
    mock_stream = {
        "info": {
            "name": ["TestStream"],
            "type": ["EEG"],
            "channel_count": ["1"],
            "nominal_srate": [str(sample_rate)],
            "source_id": ["test_source"],
            "uid": ["test_uid"],
            "effective_srate": sample_rate,
            "desc": [{"channels": [{"channel": [{"label": ["channel_1"]}]}]}],
        },
        "time_series": [[0.0]] * len(timestamps),
        "time_stamps": timestamps,
    }
    return DataStream.DataStream(stream=mock_stream)


def test_calculate_dropped_samples_no_gaps() -> None:
    """Test calculate_dropped_samples with perfect sampling (no gaps)."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate
    timestamps = [i * interval for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)
    ds.calculate_dropped_samples(onset_timestamp=0.0, offset_timestamp=0.99)

    assert ds.qc_metrics["percent_lost"] == 0.0


def test_calculate_dropped_samples_gap_in_middle() -> None:
    """Test calculate_dropped_samples with a gap in the middle of data."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Create timestamps with a 0.5 second gap at timestamp 0.5
    timestamps = [i * interval for i in range(50)]
    timestamps.extend([i * interval + 0.5 for i in range(50, 100)])

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = timestamps[-1]
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # Gap is 0.5 + interval (actual gap between samples)
    # Expected interval is 0.01, actual gap is 0.51
    # Excess = 0.51 - 0.01 = 0.5 seconds
    total_time = offset - onset
    expected_percent = (0.5 / total_time) * 100.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-5
    )


def test_calculate_dropped_samples_gap_at_start() -> None:
    """Test calculate_dropped_samples with a gap at the start."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Start sampling at 0.5 seconds instead of 0.0 (gap at start)
    start_offset = 0.5
    timestamps = [start_offset + i * interval for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = timestamps[-1]
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # Gap at start is 0.5 seconds, expected interval is 0.01
    # Excess = 0.5 - 0.01 = 0.49 seconds
    total_time = offset - onset
    expected_percent = (0.49 / total_time) * 100.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-5
    )


def test_calculate_dropped_samples_gap_at_end() -> None:
    """Test calculate_dropped_samples with a gap at the end."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Perfect sampling, but offset extends beyond last sample
    timestamps = [i * interval for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = timestamps[-1] + 0.5  # Gap of 0.5 seconds at end
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # Gap at end is 0.5 seconds, expected interval is 0.01
    # Excess = 0.5 - 0.01 = 0.49 seconds
    total_time = offset - onset
    expected_percent = (0.49 / total_time) * 100.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-5
    )


def test_calculate_dropped_samples_gaps_at_both_ends() -> None:
    """Test calculate_dropped_samples with gaps at both start and end."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Start at 0.3, end 0.3 before offset
    start_offset = 0.3
    timestamps = [start_offset + i * interval for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = timestamps[-1] + 0.3
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # Gap at start: 0.3 - 0.01 = 0.29
    # Gap at end: 0.3 - 0.01 = 0.29
    # Total excess: 0.58 seconds
    total_time = offset - onset
    expected_percent = (0.58 / total_time) * 100.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-5
    )


def test_calculate_dropped_samples_multiple_gaps() -> None:
    """Test calculate_dropped_samples with multiple gaps throughout."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Create timestamps with multiple gaps
    timestamps = [i * interval for i in range(25)]
    base = 25 * interval + 0.2
    timestamps.extend([base + i * interval for i in range(25)])
    base = base + 25 * interval + 0.3
    timestamps.extend([base + i * interval for i in range(50)])

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = timestamps[-1]
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # First gap: 0.2 + 0.01 = 0.21, excess = 0.21 - 0.01 = 0.20
    # Second gap: 0.3 + 0.01 = 0.31, excess = 0.31 - 0.01 = 0.30
    # Total excess: 0.50 seconds
    total_time = offset - onset
    expected_percent = (0.50 / total_time) * 100.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-5
    )


def test_calculate_dropped_samples_no_data_in_range() -> None:
    """Test calculate_dropped_samples with no data in the specified range."""
    sample_rate = 100.0
    timestamps = [i * 0.01 for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)

    # Request range outside of data
    with pytest.raises(ValueError, match="No data found in the specified time range."):
        ds.calculate_dropped_samples(onset_timestamp=10.0, offset_timestamp=20.0)


def test_calculate_dropped_samples_raises_negative_timestamps() -> None:
    """Test that ValueError is raised for negative timestamps."""
    sample_rate = 100.0
    timestamps = [i * 0.01 for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)

    with pytest.raises(
        ValueError, match="Onset and offset timestamps must be positive values."
    ):
        ds.calculate_dropped_samples(onset_timestamp=-1.0, offset_timestamp=1.0)


def test_calculate_dropped_samples_raises_offset_less_than_onset() -> None:
    """Test that ValueError is raised when offset <= onset."""
    sample_rate = 100.0
    timestamps = [i * 0.01 for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)

    with pytest.raises(
        ValueError, match="Offset timestamp must be greater than onset timestamp."
    ):
        ds.calculate_dropped_samples(onset_timestamp=1.0, offset_timestamp=0.5)


def test_calculate_dropped_samples_raises_offset_equals_onset() -> None:
    """Test that ValueError is raised when offset equals onset."""
    sample_rate = 100.0
    timestamps = [i * 0.01 for i in range(100)]

    ds = create_test_datastream(sample_rate, timestamps)

    with pytest.raises(
        ValueError, match="Offset timestamp must be greater than onset timestamp."
    ):
        ds.calculate_dropped_samples(onset_timestamp=1.0, offset_timestamp=1.0)


def test_calculate_dropped_samples_within_tolerance() -> None:
    """Test calculate_dropped_samples with minor variations within tolerance."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Add small variations within 5% tolerance
    timestamps = []
    for i in range(100):
        # Add small jitter within tolerance (< 5% of 0.01s interval)
        jitter = (i % 2) * 0.0003  # 3% variation
        timestamps.append(i * interval + jitter)

    ds = create_test_datastream(sample_rate, timestamps)
    ds.calculate_dropped_samples(onset_timestamp=0.0, offset_timestamp=timestamps[-1])

    # All gaps should be within tolerance, so 0% loss
    assert ds.qc_metrics["percent_lost"] == 0.0


def test_calculate_dropped_samples_single_sample() -> None:
    """Test calculate_dropped_samples with only one sample in range."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Only one sample at 0.5
    timestamps = [0.5]

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = 1.0
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # Gap at start: 0.5 - 0.0 = 0.5, excess = 0.5 - 0.01 = 0.49
    # Gap at end: 1.0 - 0.5 = 0.5, excess = 0.5 - 0.01 = 0.49
    # Total excess: 0.98 seconds out of 1.0 second window
    expected_percent = 98.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-5
    )


def test_calculate_dropped_samples_exact_boundaries() -> None:
    """Test calculate_dropped_samples when samples align exactly with boundaries."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Samples start exactly at onset and end exactly at offset
    timestamps = [i * interval for i in range(101)]  # 0.0 to 1.0

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = 1.0
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # No gaps since samples align perfectly with boundaries
    assert ds.qc_metrics["percent_lost"] == 0.0


def test_calculate_dropped_samples_large_gap() -> None:
    """Test calculate_dropped_samples with a very large gap."""
    sample_rate = 100.0
    interval = 1.0 / sample_rate

    # Create a massive gap in the middle
    timestamps = [i * interval for i in range(10)]  # 0.0 to 0.09
    timestamps.extend(
        [5.0 + i * interval for i in range(10)]
    )  # 5.0 to 5.09 (4.9 second gap)

    ds = create_test_datastream(sample_rate, timestamps)
    onset = 0.0
    offset = timestamps[-1]
    ds.calculate_dropped_samples(onset_timestamp=onset, offset_timestamp=offset)

    # Gap is ~4.91 seconds (5.0 - 0.09), expected interval is 0.01
    # Excess = 4.91 - 0.01 = 4.90 seconds
    total_time = offset - onset
    expected_percent = (4.90 / total_time) * 100.0

    assert math.isclose(
        ds.qc_metrics["percent_lost"], expected_percent, rel_tol=1e-4
    )
