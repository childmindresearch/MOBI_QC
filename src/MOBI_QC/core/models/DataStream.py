"""DataStream class module."""

import polars as pl


class DataStream:
    """Class representing a data stream.

    Attributes:
        stream_name: String representing the name of the stream assigned at collection.
        data: polars.DataFrame containing the collected data.
        variables: List of variable names in the data stream.
        data_modality: String indicating the type of data stream (e.g., EEG, ECG).
        channel_count: Integer representing the number of channels per stream.
        nominal_srate: Float indicating the sampling rate as advertised by the data
            source.
        source_id: String representing the unique identifier of the device / data
            source.
        effective_srate: Float indicating the measured sampling rate of the data stream.
        uid: String, unique ID of the stream outlet instance. Guaranteed to be
            different across multiple instantiations of the same outlet (e.g., after
            restart).
        desc: Dictionary containing extended description and metadata about the
            data stream.
        qc_metrics: Dictionary to hold quality control metrics and results.
        error: Boolean flag indicating if there was an error during processing or
            collection.

    """

    def __init__(self, stream: dict) -> None:
        """Initialize the DataStream with provided attributes from the raw stream data.

        Sets the following attributes:
            stream_name: Name of the stream assigned at collection.
            data: polars.DataFrame containing the collected data.
            variables: List of variable names in the data stream.
            data_modality: Type of data stream (e.g., EEG, ECG).
            channel_count: Number of channels per channel.
            nominal_srate: Sampling rate as advertised by the data source.
            source_id: String, unique identifier of the device / data source.
            uid: Unique ID of the stream outlet instance.
            effective_srate: Measured sampling rate of the data stream.
            desc: Extended description and metadata about the data stream.

        Args:
            stream: Raw stream data dictionary. Containing all time series and metadata.

        """
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        column_labels = [channels[i]["label"][0] for i in range(len(channels))]

        time_series_data = pl.DataFrame(
            stream["time_series"], schema=column_labels, orient="row"
        )
        full_df = time_series_data.with_columns(
            pl.Series("time_stamp", stream["time_stamps"])
        )

        self.stream_name = stream["info"]["name"][0]
        self.data = full_df.sort("time_stamp")
        self.variables = column_labels + ["time_stamp"]
        self.data_modality = stream["info"]["type"][0]
        self.channel_count = int(stream["info"]["channel_count"][0])
        self.nominal_srate = float(stream["info"]["nominal_srate"][0])
        self.source_id = stream["info"]["source_id"][0]
        self.effective_srate = stream["info"]["effective_srate"]
        self.uid = stream["info"]["uid"][0]
        self.desc = stream["info"]["desc"][0]
        self.qc_metrics: dict[str, object] = {}
        self.error = False

    def filter_time_range(
        self, onset_timestamp: float, offset_timestamp: float
    ) -> None:
        """Filter DataStream.data attribute.

        Reassign the DataStream.data attribute to only include data within
        a specified time range, based on LSL timestamps.
        Recalculates sampling rate based on filtered data.

        DataStream.data will be empty if onset_timestamp is greater than the
        last value of data['time_stamp'], if offset_timestamp is less than the
        first value of data['time_stamp'], or if onset_timestamp and offset_timestamp
        are in between two values of data['time_stamp'].

        Args:
            onset_timestamp: start time (seconds) for filtering the data
            offset_timestamp: end time (seconds) for filtering the data

        Raises:
            ValueError: If offset_timestamp is less than or equal to onset_timestamp.
                        If either onset_timestamp or offset_timestamp is negative.
        """
        if offset_timestamp < 0 or onset_timestamp < 0:
            raise ValueError("Onset and offset timestamps must be positive values.")

        if offset_timestamp <= onset_timestamp:
            raise ValueError("Offset timestamp must be greater than onset timestamp.")

        self.data = self.data.filter(
            (pl.col("time_stamp") >= onset_timestamp)
            & (pl.col("time_stamp") <= offset_timestamp)
        )

        if len(self.data) >= 2:
            time_stamp_diff = float(
                self.data.select(pl.col("time_stamp").diff()).mean().item()
            )
            self.effective_srate = 1 / time_stamp_diff
        else:
            self.effective_srate = 0


    def calculate_dropped_samples(
        self, onset_timestamp: float, offset_timestamp: float
    ) -> float:
        """Calculate the percentage of time lost due to dropped samples.

        This method analyzes gaps between consecutive samples in the data stream
        and calculates the total time lost due to gaps exceeding the expected
        sampling interval. Gaps at the beginning (onset to first sample) and end
        (last sample to offset) are automatically included by adding virtual
        boundary timestamps.

        Args:
            onset_timestamp: Start time (seconds) for the analysis window.
            offset_timestamp: End time (seconds) for the analysis window.

        Returns:
            None. Updates the qc_metrics dictionary with the percentage of time lost
            due to dropped samples under the key "percent_lost".

        Raises:
            ValueError: If offset_timestamp <= onset_timestamp or if either is negative.
        """
        # Validate inputs
        if offset_timestamp < 0 or onset_timestamp < 0:
            raise ValueError("Onset and offset timestamps must be positive values.")

        if offset_timestamp <= onset_timestamp:
            raise ValueError("Offset timestamp must be greater than onset timestamp.")

        # Filter data to the specified time range
        filtered_data = self.data.filter(
            (pl.col("time_stamp") >= onset_timestamp)
            & (pl.col("time_stamp") <= offset_timestamp)
        )

        if filtered_data.height < 1:
            raise ValueError("No data found in the specified time range.")

        # Calculate expected interval based on nominal sampling rate
        expected_interval = 1.0 / self.nominal_srate

        # Add tolerance (5%) to account for minor timing variations
        tolerance_threshold = expected_interval * 1.05

        # Extract timestamps and add virtual boundaries at onset and offset
        # This automatically handles gaps at the start and end
        timestamps = filtered_data.select("time_stamp").to_series().to_list()
        extended_timestamps = [onset_timestamp] + timestamps + [offset_timestamp]

        # Create DataFrame with extended timestamps
        df_extended = pl.DataFrame({"time_stamp": extended_timestamps})

        # Calculate time differences between consecutive timestamps
        df_with_diffs = df_extended.with_columns(
            pl.col("time_stamp").diff().alias("time_diff")
        )

        # Filter for gaps exceeding the tolerance threshold
        # Calculate excess time for each gap (gap - expected_interval)
        # Sum all excess times
        gaps_excess = (
            df_with_diffs.filter(pl.col("time_diff") > tolerance_threshold)
            .select((pl.col("time_diff") - expected_interval).alias("excess"))
            .sum()
            .item()
        )

        # Handle None case when no gaps are found
        total_lost_time = float(gaps_excess) if gaps_excess is not None else 0.0

        # Calculate total time window
        total_time = offset_timestamp - onset_timestamp

        # Calculate percentage of time lost
        self.qc_metrics["percent_lost"] = (total_lost_time / total_time) * 100.0

        return 

        