from librosa import stream
import pyxdf  # noqa: D100
import os
import platform
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
from stim_correction import trigger_recovery

class BaseProcessor:
    """Base class for processing XDF data."""

    def __init__(self) -> None:
        """Initialize the BaseProcessor with default attributes."""
        print('Base Processor initialized...')
        self.stim_df = None
        self.eeg_df = None
        self.physio_df = None
        self.eye_df = None
        self.video_df = None
        self.mic_df = None
        self.xdf_path = None
        self.subject_id = None
        self.events = None
        return

    def load_data(self, xdf_path: str | None = None, 
                  stream_name: str | list | None = None, 
                  events: dict | None = None) -> None:
        """Load the XDF data.

        Args:
            xdf_path (str | None): Path to the XDF file. Defaults to None.
            stream_name (str | list | None): Name of the specific stream to load. Defaults to None.
            events (dict | None)
        """
        if xdf_path is None:
            raise ValueError("XDF path must be provided")
        if not isinstance(xdf_path, str):
            raise TypeError("XDF path must be a string")
        if not xdf_path.endswith('.xdf'):
            raise ValueError("XDF path must end with '.xdf'")   
        else:
            self.xdf_path = xdf_path
            self.subject_id = xdf_path.split('/')[-1].split('.')[0]
        if stream_name is not None and not isinstance(stream_name, str | list):
            raise TypeError("Stream name must be a string")
        if stream_name is None:
            print(f"Loading all streams from {self.xdf_path}...")
            self.data, _ = pyxdf.load_xdf(self.xdf_path, verbose=False)
            self.streams = [stream['info']['name'][0] for stream in self.data]
            # print(f"Streams loaded: {[x for x in self.streams]}")
        else:
            self.data, _ = pyxdf.load_xdf(self.xdf_path, 
                                          select_streams=[{'name': x} for x in stream_name],
                                          verbose=False)
            self.streams = stream_name
            # load stim here
            if events is not None:
                if not isinstance(events, dict):
                    raise TypeError("Events must be a dictionary")
                self.events = events
            else:
                raise ValueError("Events dictionary must be provided")

            stim_df = pd.DataFrame({'trigger': [x[0] for x in stream['time_series']], 
                                    'time_stamp': stream['time_stamps']}) 
                    stim_df['event'] = stim_df['trigger'].apply(lambda x: self.events[x] if x in self.events.keys() else 'non-trigger-event')
                    # specific to CUNY project
                    dt = datetime.strptime(self.get_collection_date(), '%Y-%m-%d %H:%M:%S')
                    # check if date after 03/25/2025
                    if (dt > datetime(2025, 3, 25)) & (dt < datetime(2025, 5, 23)):
                        print('trigger recovery')
                        stim_df = trigger_recovery(stim_df, self.xdf_path)
                    self.stim_df = stim_df


        return
    
    def get_collection_date(self) -> str:

        """Get the collection date of the XDF file.
        
        Returns:
            str: Collection date in 'YYYY-MM-DD HH:MM:SS' format.
        """
        if not hasattr(self, 'xdf_path'):
            raise AttributeError("XDF path is not set. Please load data first.")
        else:
            if platform.system() == 'Windows':
                    return datetime.fromtimestamp(os.path.getctime(self.xdf_path))
            else:
                # On Unix, getctime returns the last metadata change — not creation time
                stat = os.stat(self.xdf_path)
                try:
                    return datetime.fromtimestamp(stat.st_birthtime).strftime('%Y-%m-%d %H:%M:%S')
                except AttributeError:
                    # Fallback: use modification time instead
                    return datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

    def get_sampling_rate(self, df):
        effective_sampling_rate = 1 / (df.lsl_time_stamp.diff().median())
        return effective_sampling_rate
    
    def run_behavior_qc(self):
        behavior_qc(xdf_filename, self.stim_df)




    def 



