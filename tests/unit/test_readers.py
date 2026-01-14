"""Unit tests for the MOBI_QC.io.readers module.

This module contains tests for the read_xdf function, including:
- Validation of input parameters (path, stream_names)
- File existence and extension checks
- Stream selection and loading behavior
"""

import pathlib

import pytest

from MOBI_QC.io.readers import readers


def test_raises_value_error_when_extension_is_not_xdf(tmp_path: pathlib.Path) -> None:
    """Test that read_xdf raises ValueError when file extension is not .xdf."""
    bad_path = tmp_path / "file.notxdf"
    bad_path.write_text("dummy")
    with pytest.raises(ValueError, match="XDF path must end with '.xdf'"):
        readers.read_xdf(str(bad_path))


def test_raises_file_not_found_for_missing_file(tmp_path: pathlib.Path) -> None:
    """Test that read_xdf raises FileNotFoundError when the file does not exist."""
    missing = tmp_path / "does_not_exist.xdf"

    with pytest.raises(FileNotFoundError, match="XDF file not found"):
        readers.read_xdf(str(missing))


def test_reads_all_streams_when_stream_names_is_none(
    sample_xdf_file: pathlib.Path,
) -> None:
    """Test that read_xdf reads all streams when stream_names is None."""
    expected_stream_names = [
        "EventMarkerStream",
        "PhysioStream",
        "GazeStream",
        "CameraFrameStream",
        "EEGStream",
        "MicrophoneStream"
    ]
    streams = readers.read_xdf(sample_xdf_file, stream_names=None)
    stream_names = [streams[i]["info"]["name"][0] for i in range(len(streams))]
    assert stream_names == expected_stream_names


def test_reads_selected_streams_and_builds_select_streams(
    sample_xdf_file: pathlib.Path,
) -> None:
    """Test that read_xdf reads selected streams and builds select_streams correctly."""
    requested_names = ["EventMarkerStream", "GazeStream"]
    streams = readers.read_xdf(sample_xdf_file, stream_names=requested_names)
    stream_names = [streams[i]["info"]["name"][0] for i in range(len(streams))]
    assert stream_names == requested_names
