import pandas as pd
import pyxdf
import tarfile
from io import BytesIO
import os
import platform

import numpy as np
import sounddevice as sd
from glob import glob
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from stim_correction import trigger_recovery


def get_collection_date(xdf_filename:str):
    if platform.system() == 'Windows':
            return datetime.datetime.fromtimestamp(os.path.getctime(xdf_filename))
    else:
        # On Unix, getctime returns the last metadata change — not creation time
        stat = os.stat(xdf_filename)
        try:
            return datetime.datetime.fromtimestamp(stat.st_birthtime).strftime('%Y-%m-%d %H:%M:%S')
        except AttributeError:
            # Fallback: use modification time instead
            return datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

def import_webcam_data(xdf_filename:str):    
    cam_data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'WebcamStream'}], verbose=False)
    frame_nums = [int(i[0]) for i in cam_data[0]['time_series']]
    time_pre = [float(i[1]) for i in cam_data[0]['time_series']]
    time_evnt_ms = [float(i[2]) for i in cam_data[0]['time_series']]
    time_post = [float(i[3]) for i in cam_data[0]['time_series']]

    cam_df = pd.DataFrame({'frame_num': frame_nums, 
                        'time_pre': time_pre, 
                        'cap_time_ms': time_evnt_ms,
                        'time_post': time_post,
                        'lsl_time_stamp': cam_data[0]['time_stamps']})

    cam_df['frame_time_sec'] = (cam_df.cap_time_ms - cam_df.cap_time_ms[0])/1000
    cam_df['lsl_time_sec'] = (cam_df.lsl_time_stamp - cam_df.lsl_time_stamp[0]) *1000
    return cam_df


def import_physio_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'OpenSignals'}], verbose = False)
    column_labels = [data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data[0]['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data[0]['time_series'], columns=column_labels)
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_mic_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'AudioCapture'}], verbose = False)
    df = pd.DataFrame(data[0]['time_series'], columns=['int_array'])
    df['bytestring'] = df['int_array'].apply(lambda x: np.array(x).tobytes())
    df['duration'] = (data[0]['time_stamps'] - data[0]['time_stamps'][0])/data[0]['info']['effective_srate']
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_video_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'video'}], verbose = False)
    frame_nums = [int(i[0]) for i in data[0]['time_series']]
    time_pre = [float(i[1]) for i in data[0]['time_series']]
    time_evnt_ms = [float(i[2]) for i in data[0]['time_series']]
    time_post = [float(i[3]) for i in data[0]['time_series']]
    df = pd.DataFrame({'frame_num': frame_nums, 
                        'time_pre': time_pre, 
                        'cap_time_ms': time_evnt_ms,
                        'time_post': time_post,
                        'lsl_time_stamp': data[0]['time_stamps']})

    df['frame_time_sec'] = (df.cap_time_ms - df.cap_time_ms[0])/1000
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_et_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'ET'}], verbose = False)
    column_labels = [data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data[0]['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data[0]['time_series'], columns=column_labels)
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    df['diff'] = df.lsl_time_stamp.diff()
    return df

def import_eeg_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'EEG'}], verbose = False)
    ch_names = [f"E{i+1}" for i in range(data[0]['time_series'].shape[1])]
    df = pd.DataFrame(data[0]['time_series'], columns=ch_names) # index=data[0]['time_stamps']
    df['lsl_time_stamp'] = data[0]['time_stamps']
    #df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_stim_data(xdf_filename:str):
    '''
    Get the stimuli dataframe from the xdf file.
    
    Args:
        xdf_filename (str): The xdf file to get the stimuli from.
    '''
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name':'Stimuli_Markers'}], verbose = False)
    stim_df = pd.DataFrame(data[0]['time_series'])
    stim_df.rename(columns={0: 'trigger'}, inplace=True)

    events = {
        200: 'Onset_Experiment',
        10: 'Onset_RestingState',
        11: 'Offset_RestingState',
        500: 'Onset_StoryListening',
        501: 'Offset_StoryListening',
        100: 'Onset_10second_rest',
        101: 'Offset_10second_rest', 
        20: 'Onset_CampFriend',
        21: 'Offset_CampFriend',
        30: 'Onset_FrogDissection',
        31: 'Offset_FrogDissection',
        40: 'Onset_DanceContest',
        41: 'Offset_DanceContest',
        50: 'Onset_ZoomClass',
        51: 'Offset_ZoomClass',
        60: 'Onset_Tornado',
        61: 'Offset_Tornado',
        70: 'Onset_BirthdayParty',
        71: 'Offset_BirthdayParty',
        300: 'Onset_subjectInput',
        301: 'Offset_subjectInput',
        302: 'Onset_FavoriteStory',
        303: 'Offset_FavoriteStory',
        304: 'Onset_WorstStory',
        305: 'Offset_WorstStory',
        400: 'Onset_impedanceCheck',
        401: 'Offset_impedanceCheck',
        80: 'Onset_SocialTask',
        81: 'Offset_SocialTask',
        201: 'Offset_Experiment',
    }

    story_onsets = [20, 30, 40, 50, 60, 70]

    # relabel the event if the trigger is in the events dictionary, else if 
    stim_df['event'] = stim_df['trigger'].apply(lambda x: events[x] if x in events.keys() else 'Bx_input')

    # relabel the event as a psychopy timestamp if the trigger is greater than 5 digits
    stim_df.loc[stim_df.trigger.astype(str).str.len() > 5, 'event'] = 'psychopy_time_stamp'
    stim_df['lsl_time_stamp'] = data[0]['time_stamps']
    #stim_df['time'] = (data[0]['time_stamps'] - data[0]['time_stamps'][0])

    dt = datetime.datetime.fromtimestamp(stim_df.loc[stim_df.event == "psychopy_time_stamp", "trigger"].to_list()[0])#.strftime('%Y-%m-%d %H:%M:%S')
    # check if date after 03/25/2025
    
    if (dt > datetime.datetime(2025, 3, 25)) & (dt < datetime.datetime(2025, 5, 23)):
        print('trigger recovery')
        stim_df = trigger_recovery(stim_df, xdf_filename)
    
    return stim_df

def get_event_data(event, df, stim_df):
    """
    Get the data from a given data modality that corresponds to the event in the stimuli dataframe.
    
    Args:
        event (str): The event to get the data for.
        df (pd.DataFrame): The dataframe containing the timeseries along with a column for lsl timestamps.
        stim_df (pd.DataFrame): The stimuli dataframe containing the eventa mapped to lsl timestamps.
    
    Returns:
        pd.DataFrame: The  data corresponding to the event.
        """
    new_df = df.loc[(df['lsl_time_stamp'] >= stim_df.loc[stim_df['event'] == 'Onset_'+event, 'lsl_time_stamp'].values[0]) & 
                  (df['lsl_time_stamp'] <= stim_df.loc[stim_df['event'] == 'Offset_'+event, 'lsl_time_stamp'].values[0])].copy().reset_index(drop = True)
    return new_df

def load_xdf_from_zip(path_to_zip):  
    # Path to the tar.gz file
    tar_gz_file_path = path_to_zip # Path to the tar.gz file

    # Open the tar.gz file
    with tarfile.open(tar_gz_file_path, 'r:gz') as tar:
        file_list = tar.getnames() # List all files in the tar.gz
        file_name = [x for x in file_list if os.path.splitext(x)[1] == '.xdf'][0] # Read a specific file from the tar.gz
        file = tar.extractfile(file_name)
        file_content = file.read()
        data, info = pyxdf.load_xdf(BytesIO(file_content))
        #streams_collected = [stream['info']['name'][0] for stream in data]        
        #print(streams_collected)
    return data, info

def get_sampling_rate(df):
    effective_sampling_rate = 1 / (df.lsl_time_stamp.diff().median())
    return effective_sampling_rate

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass