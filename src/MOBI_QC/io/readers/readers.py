"""This module contains functions to read MOBI xdf files."""

import pathlib
from typing import Optional

import pyxdf


def read_xdf(
    xdf_path: str | pathlib.Path, stream_names: Optional[list[str]] = None
) -> list:
    """Reads a MOBI xdf file and returns a list containing the data streams.

    Args:
    ----------
    xdf_path :
        Path to the MOBI xdf file.
    stream_names :
        List of stream names to read from the xdf file. If None, all streams are read.

    Returns:
    -------
    list
        A list of data streams contained in the xdf file.

    Raises:
    ------
    ValueError
        If xdf_path does not end with '.xdf'.
    TypeError
        If xdf_path is not a string or pathlib.Path, or if stream_names is not a list of strings.
    FileNotFoundError
        If the xdf file does not exist.
    """
    xdf_path = pathlib.Path(xdf_path)

    if xdf_path.suffix != ".xdf":
        raise ValueError("XDF path must end with '.xdf'")

    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    if stream_names is None:
        streams, _ = pyxdf.load_xdf(xdf_path)

    else:
        streams, _ = pyxdf.load_xdf(
            xdf_path,
            select_streams=[{"name": name} for name in stream_names],
            verbose=False,
        )

    return streams
