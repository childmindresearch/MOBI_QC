"""BaseProcessor class module."""

import pathlib
from typing import Optional

from MOBI_QC.io.readers import readers


class BaseProcessor:
    """Base class for processing XDF data.

    Attributes:
        xdf_path: pathlib.Path to the XDF file.
        subject_id: string identifying the subject extracted from the file name.
        raw_data: list of raw data streams read from the XDF file.
        stream_names: list of names of the data streams to process.
    """

    def __init__(
        self, xdf_path: str | pathlib.Path, stream_names: Optional[list[str]] = None
    ) -> None:
        """Initialize the BaseProcessor with default attributes.

        Args:
            xdf_path: Path to the XDF file.
            stream_names: Optional list of stream names to process.
        """
        self.xdf_path = pathlib.Path(xdf_path)
        self.subject_id = self.xdf_path.stem.split("_")[0]
        self.raw_data = readers.read_xdf(xdf_path, stream_names)
        self.stream_names = stream_names or [
            stream["info"]["name"][0] for stream in self.raw_data
        ]
