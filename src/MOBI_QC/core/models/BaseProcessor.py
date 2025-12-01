"""BaseProcessor class module."""

import pathlib
from typing import Optional

from MOBI_QC.io.readers import readers


class BaseProcessor:
    """Base class for processing XDF data."""

    def __init__(
        self, xdf_path: str | pathlib.Path, stream_names: Optional[list[str]] = None
    ) -> None:
        """Initialize the BaseProcessor with default attributes."""
        self.xdf_path = pathlib.Path(xdf_path)
        self.subject_id = self.xdf_path.stem.split("_")[0]
        self.raw_data = readers.read_xdf(xdf_path, stream_names)
        if stream_names is None:
            self.stream_names = [stream["info"]["name"][0] for stream in self.raw_data]
        else:
            self.stream_names = stream_names
