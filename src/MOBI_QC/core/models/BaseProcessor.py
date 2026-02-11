"""BaseProcessor class module."""

import datetime
import os
import pathlib
import platform
from typing import Optional

from MOBI_QC.core.models.DataStream import DataStream
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

    def format_data(self) -> None:
        """Format the raw data.

        This method organizes the streams in the raw data into DataStream
        objects and assigns them as attributes of the processor instance.
        """
        for stream in self.raw_data:
            ds = DataStream(stream=stream)
            setattr(self, stream["info"]["type"][0], ds)

    def get_collection_date(self) -> str:
        """Extract the collection date from the XDF file name.

        Returns:
            A string representing the collection date in 'YYYY-MM-DD HH:MM:SS' format.
        """
        if platform.system() == "Windows":
            timestamp = datetime.datetime.fromtimestamp(os.path.getctime(self.xdf_path))
        else:
            stat = os.stat(self.xdf_path)
            try:
                timestamp = datetime.datetime.fromtimestamp(stat.st_birthtime)

            except AttributeError:
                # Fallback: use modification time instead
                timestamp = datetime.datetime.fromtimestamp(stat.st_mtime)

        self.collection_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return self.collection_date
