import pathlib
import types

import pytest
from src.MOBI_QC.io.readers import read_xdf


def test_raises_value_error_when_xdf_path_is_none():
    with pytest.raises(ValueError, match="XDF path must be provided"):
        read_xdf(None)


def test_raises_type_error_when_xdf_path_wrong_type():
    with pytest.raises(TypeError, match="XDF path must be a string or pathlib.Path"):
        read_xdf(123)  # int is invalid


def test_raises_value_error_when_extension_is_not_xdf(tmp_path):
    bad_path = tmp_path / "file.notxdf"
    bad_path.write_text("dummy")
    with pytest.raises(ValueError, match="XDF path must end with '.xdf'"):
        read_xdf(str(bad_path))


def test_raises_type_error_when_stream_names_not_list(tmp_path):
    good_path = tmp_path / "file.xdf"
    good_path.write_text("dummy")
    with pytest.raises(TypeError, match="Stream names must be a list of strings"):
        read_xdf(str(good_path), stream_names="not-a-list")


def test_raises_file_not_found_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.xdf"
    # do NOT create the file
    with pytest.raises(FileNotFoundError, match="XDF file not found"):
        read_xdf(str(missing))


def test_reads_all_streams_when_stream_names_is_none(tmp_path, monkeypatch):
    # create a dummy .xdf file so pathlib.Path(...).exists() is True
    xdf_file = tmp_path / "sample.xdf"
    xdf_file.write_text("dummy content")

    # Prepare a fake pyxdf module with a load_xdf function
    captured = {}

    def fake_load_xdf(path_arg, *args, **kwargs):
        # Capture the exact arguments passed
        captured['path_arg'] = path_arg
        captured['args'] = args
        captured['kwargs'] = kwargs
        # return streams and meta (pyxdf.load_xdf returns (streams, header))
        return (["streamA", "streamB"], {"header": "meta"})

    fake_pyxdf = types.SimpleNamespace(load_xdf=fake_load_xdf)

    # Patch the pyxdf object used inside the module under test
    monkeypatch.setattr(read_xdf.__module__, "pyxdf", fake_pyxdf)

    streams = read_xdf(xdf_file)

    # Assertions
    assert streams == ["streamA", "streamB"]
    # The path passed into load_xdf should be a pathlib.Path (module converts to Path)
    assert isinstance(captured['path_arg'], pathlib.Path)
    assert captured['path_arg'].name == "sample.xdf"
    # When stream_names is None we expect no select_streams kwarg
    assert "select_streams" not in captured['kwargs']


def test_reads_selected_streams_and_builds_select_streams(tmp_path, monkeypatch):
    xdf_file = tmp_path / "selected_sample.xdf"
    xdf_file.write_text("dummy content")

    captured = {}

    def fake_load_xdf(path_arg, *args, **kwargs):
        captured['path_arg'] = path_arg
        captured['args'] = args
        captured['kwargs'] = kwargs
        return (["selected_stream"], {"header": "meta"})

    fake_pyxdf = types.SimpleNamespace(load_xdf=fake_load_xdf)
    monkeypatch.setattr(read_xdf.__module__, "pyxdf", fake_pyxdf)

    requested_names = ["EEG", "ECG"]
    streams = read_xdf(xdf_file, stream_names=requested_names)

    # Basic return value check
    assert streams == ["selected_stream"]

    # Ensure select_streams kwarg was passed and is correctly formatted
    assert "select_streams" in captured['kwargs']
    select_streams = captured['kwargs']["select_streams"]
    assert isinstance(select_streams, list)
    assert select_streams == [{"name": "EEG"}, {"name": "ECG"}]

