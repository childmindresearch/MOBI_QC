"""DataStream class module."""
import polars as pl


class DataStream:
    """Class representing a data stream.
    
    Attributes:
        stream_name: String representing the name of the stream assigned at collection.
        data: polars.DataFrame containing the collected data.
        variables: List of variable names in the data stream.
        data_type: String indicating the type of data stream (e.g., EEG, ECG).
        channel_count: Integer representing the number of channels per channel.
        nominal_srate: Float indicating the sampling rate as advertised by the data
            source.
        source_id: String representing the unique identifier of the device / data
            source.
        effective_srate: Float indicating the measured sampling rate of the data stream.
        uid: String, unique ID of the stream outlet instance. Guaranteed to be
            different across multiple instantiations of the same outlet (e.g., after
            restart).
        created_at: String, timestamp when the stream was first created (as
            determined via lsl::local_clock()).
        desc: Dictionary containing extended description and metadata about the
            data stream.
        qc: Dictionary to hold quality control metrics and results.
        error: Boolean flag indicating if there was an error during processing or 
            collection.
        
    """
    def __init__(self, stream_name: str, data: pl.DataFrame, variables: list,
                 data_type: str, channel_count: int, nominal_srate: float, 
                 source_id: str, uid: str, created_at: str, effective_srate: float, 
                 desc: dict) -> None:
        """Initialize the DataStream with provided attributes.
        
        Args:
            stream_name: Name of the stream assigned at collection.
            data: polars.DataFrame containing the collected data.
            variables: List of variable names in the data stream.
            data_type: Type of data stream (e.g., EEG, ECG).
            channel_count: Number of channels per channel.
            nominal_srate: Sampling rate as advertised by the data source.
            source_id: String, unique identifier of the device / data source.
            uid: Unique ID of the stream outlet instance.
            created_at: Timestamp when the stream was first created.
            effective_srate: Measured sampling rate of the data stream.
            desc: Extended description and metadata about the data stream.
        """
        self.stream_name = stream_name
        self.data = data
        self.variables = variables
        self.data_type = data_type
        self.channel_count = channel_count
        self.nominal_srate = nominal_srate
        self.source_id = source_id
        self.effective_srate = effective_srate
        self.uid = uid
        self.created_at = created_at
        self.desc = desc
        self.qc = {}
        self.error = False