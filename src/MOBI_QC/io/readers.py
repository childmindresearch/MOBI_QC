"""This module contains functions to read MOBI xdf files."""
import pyxdf
import pathlib
from typing import Optional


def read_xdf(xdf_path: str | pathlib.Path, stream_names: Optional[list[str]] = None) -> list:
    """Reads a MOBI xdf file and returns the data streams.

    Parameters
    ----------
    xdf_path : str
        Path to the MOBI xdf file.
    stream_name : list[str], optional
        List of stream names to read from the xdf file. If None, all streams are read.

    Returns
    -------
    list
        A list of data streams contained in the xdf file.
    """
    if xdf_path is None:
        raise ValueError("XDF path must be provided")
    if not isinstance(xdf_path, str):
        raise TypeError("XDF path must be a string")
    if not xdf_path.endswith('.xdf'):
        raise ValueError("XDF path must end with '.xdf'")  
    
    if stream_names is not None and not isinstance(stream_names, list):
        raise TypeError("Stream names must be a list of strings")
    

    xdf_path = pathlib.Path(xdf_path)
    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")


    if stream_names is None:
        streams, _ = pyxdf.load_xdf(xdf_path)

    else:
        streams, _ = pyxdf.load_xdf(xdf_path, 
                                    select_streams=[{"name": name} for name in stream_names],
                                    verbose=False)
        
    return streams


if __name__ == "__main__":
    pass