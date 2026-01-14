"""BaseProcessor class module."""

import pathlib
from typing import Optional

import polars as pl

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

        This method organized the streams in the raw data into DataStream
        objects and assigns them as attributes of the processor instance.
        """
        for stream in self.raw_data:
            channels = stream['info']['desc'][0]['channels'][0]['channel']
            column_labels = [
                channels[i]['label'][0] for i in range(len(channels))
            ]
            df = pl.DataFrame(stream['time_series'], schema=column_labels, orient="row")
            df = df.with_columns(pl.Series('time_stamp', stream['time_stamps']))
            
            ds = DataStream(
                stream_name=stream['info']['name'][0],
                data=df,
                variables=column_labels + ['time_stamp'],
                data_modality=stream['info']['type'][0],
                channel_count=stream['info']['channel_count'][0],
                nominal_srate=stream['info']['nominal_srate'][0],
                source_id=stream['info']['source_id'][0],
                uid=stream['info']['uid'][0],
                effective_srate=stream['info']['effective_srate'],
                desc=stream['info']['desc'][0],
            )
           
            setattr(self, stream['info']['type'][0], ds)        
            
            