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

        else:
            self.data, _ = pyxdf.load_xdf(self.xdf_path, 
                                          select_streams=[{'name': x} for x in stream_name],
                                          verbose=False)
            self.streams = stream_name

        if events is not None:
            if not isinstance(events, dict):
                raise TypeError("Events must be a dictionary")
            self.events = events
        else:
            raise ValueError("Events dictionary must be provided")

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

    def format_data(self) -> None:
        """Format the loaded data into pandas DataFrames."""
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Please load data first.")

        for stream in tqdm(self.data, desc="Formatting data", unit="stream"):
            s_name = stream['info']['name'][0]
            if s_name in ['Stimulus', 'Stimuli_Markers']:
                if not hasattr(self, 'events'):
                    raise AttributeError("Events dictionary not loaded. Please load events dictionary first.")
                else:
                    df = pd.DataFrame({'trigger': [x[0] for x in stream['time_series']], 
                                    'time_stamp': stream['time_stamps']}) 
                    df['event'] = df['trigger'].apply(lambda x: self.events[x] if x in self.events.keys() else 'non-trigger-event')
                    # specific to CUNY project
                    dt = datetime.strptime(self.get_collection_date(), '%Y-%m-%d %H:%M:%S')
                    # check if date after 03/25/2025
                    if (dt > datetime(2025, 3, 25)) & (dt < datetime(2025, 5, 23)):
                        print('trigger recovery')
                        df = trigger_recovery(df, self.xdf_path)
                    self.stim_df = df
    
            elif s_name in ['EEG', 'EGI NetAmp 0']:
                ch_names = [f"E{i+1}" for i in range(stream['time_series'].shape[1])]
                df = pd.DataFrame(stream['time_series'], columns=ch_names) # index=stream['time_stamps']
                df['time_stamp'] = stream['time_stamps']
                self.eeg_df = df

            elif s_name in ['Microphone']:
                df = pd.DataFrame({
                    "int_array": stream['time_series'].flatten(),
                    "time_stamp": stream['time_stamps']})
                self.mic_df = df

            elif s_name in ['EyeTracking', 'Tobii']:
                column_labels = [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(stream['info']['desc'][0]['channels'][0]['channel']))]
                df = pd.DataFrame(stream['time_series'], columns=column_labels)
                df['time_stamp'] = stream['time_stamps']
                self.eye_df = df

            elif s_name in ['Video', 'WebcamStream']:
                ts_array = np.array(stream['time_series'], dtype=float)
                df = pd.DataFrame({
                    'frame_num': ts_array[:, 0].astype(int),
                    'time_pre': ts_array[:, 1],
                    'cap_time_ms': ts_array[:, 2],
                    'time_post': ts_array[:, 3],
                    'time_stamp': stream['time_stamps']
                })
                self.video_df = df  
            
            elif s_name in ['Physio', 'OpenSignals']:
                column_labels = [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(stream['info']['desc'][0]['channels'][0]['channel']))]
                df = pd.DataFrame(stream['time_series'], columns=column_labels)
                df['time_stamp'] = stream['time_stamps']
                self.physio_df = df

    # Additional methods specific to EEG processing can be added here.

['OpenSignals',
 'Stimuli_Markers',
 'WebcamStream',
 'Tobii',
 'Microphone']

if __name__ == "__main__":
    pass