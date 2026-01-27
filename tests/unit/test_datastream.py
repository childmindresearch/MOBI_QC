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
