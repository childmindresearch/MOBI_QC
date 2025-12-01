"""BaseProcessor class module."""
import datetime
import os
import pathlib
import platform
from typing import Optional

from MOBI_QC.io.readers import readers


class BaseProcessor:
    """Base class for processing XDF data."""

    def __init__(
        self, xdf_path: str | pathlib.Path,
        stream_names: Optional[list[str]] = None
    )-> None:
        """Initialize the BaseProcessor with default attributes."""
        self.xdf_path = pathlib.Path(xdf_path)
        self.subject_id = self.xdf_path.stem.split("_")[0]
        self.collection_date = self.get_collection_date()
        self.raw_data = readers.read_xdf(xdf_path, stream_names)
        if stream_names is None:
            self.stream_names = [
                stream["info"]["name"][0] for stream in self.raw_data
            ]
        else:
            self.stream_names = stream_names

    def get_collection_date(self) -> str:
        """Extract collection date from the file name.

        Returns:
        ________    
            str: The collection date in 'YYYYMMDD' format.
        """
        if platform.system() == 'Windows':
                return datetime.datetime.fromtimestamp(
                    os.path.getctime(self.xdf_path)
                )
        else:
            stat = os.stat(self.xdf_path)
            try:
                self.collection_date = datetime.datetime.fromtimestamp(
                    stat.st_birthtime
                ).strftime('%Y-%m-%d %H:%M:%S')
                return self.collection_date
            except AttributeError:
                return datetime.datetime.fromtimestamp(
                    stat.st_mtime
                ).strftime('%Y-%m-%d %H:%M:%S')
    