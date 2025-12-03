"""Unit tests for the DataStream class."""

import pathlib

import polars as pl

from MOBI_QC.core.models import DataStream
from MOBI_QC.io.readers import readers


def test_datastream_initialization(sample_xdf_file: pathlib.Path) -> None:
    """Test DataStream initialization and attribute assignments."""
    stream = readers.read_xdf(xdf_path=sample_xdf_file, stream_names=["GazeStream"])[0]

    info = stream["info"]
    channels = info["desc"][0]["channels"][0]["channel"]
    column_labels = [channel["label"][0] for channel in channels]
    df = pl.DataFrame(stream["time_series"], schema=column_labels).with_columns(
        pl.Series("time_stamps", stream["time_stamps"])
    )

    ds = DataStream.DataStream(
        stream_name=info["name"][0],
        data=df,
        variables=column_labels,
        data_type=info["type"][0],
        channel_count=int(info["channel_count"][0]),
        nominal_srate=float(info["nominal_srate"][0]),
        source_id=info["source_id"][0],
        uid=info["uid"][0],
        created_at=info["created_at"][0],
        effective_srate=float(info["effective_srate"]),
        desc=dict(info["desc"][0]),
    )

    assert isinstance(ds.stream_name, str)
    assert isinstance(ds.data, pl.DataFrame)
    assert isinstance(ds.variables, list)
    assert isinstance(ds.data_type, str)
    assert isinstance(ds.channel_count, int)
    assert isinstance(ds.nominal_srate, float)
    assert isinstance(ds.source_id, str)
    assert isinstance(ds.uid, str)
    assert isinstance(ds.created_at, str)
    assert isinstance(ds.effective_srate, float)
    assert isinstance(ds.desc, dict)
    assert hasattr(ds, "qc")
    assert isinstance(ds.qc, dict)
    assert hasattr(ds, "error")
    assert not ds.error
