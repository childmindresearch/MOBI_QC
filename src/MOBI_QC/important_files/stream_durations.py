import pandas as pd
import os
import numpy as np
import datetime
from utils import *


# get durations of certain experiment arm
def get_durations(xdf_path: str, 
    task: str, 
    stim_df: pd.DataFrame,
    df_map: dict, 
    error_map: dict
    ) -> pd.DataFrame:
    
    """
    Get the durations of each data stream and compare to their expected duration, given an experiment arm, where the expected duration is calculated from the LSL timestamps of the stimulus markers.
    
    Args:
        xdf_path (str): The path to the xdf file.
        task (str): The part of the experiment to view durations. Can be one of "Experiment", 
            "RestingState", "StoryListening", "SocialTask", or any one of the stories ('BirthdayParty', 
            'ZoomClass', 'Tornado', 'FrogDissection', 'DanceContest', 'CampFriend')
        stim_df (pd.DataFrame): The stimuli dataframe containing the events mapped to lsl timestamps.
        df_map (dict): Contains dataframes for each data modality, loaded through import_modality_data functions in utils.
        error_map (dict): Contains booleans for each data modality indicating error. 
    
    Returns:
        pd.DataFrame: The durations of each stream in seconds and mm:ss and the percent that that duration 
            comprised of the length of that experiment arm.
    """    
    durations_dict = {}
    durations_dict['event'] = task
    streams = list(df_map.keys())

    for stream in streams: 
        duration_name = f'{stream}_duration'
        mmss_name = f'{stream}_mm:ss'
        percent_name = f'{stream}_percent'
        durations_dict[duration_name] = float('nan')
        durations_dict[mmss_name] = ''
        durations_dict[percent_name] = float('nan')
    
    if error_map['behavior']:
        return durations_dict

    # find expected duration (stim lsl_time_stamp length of experiment part)
    exp_start = stim_df.loc[stim_df.event == 'Onset_'+task, 'lsl_time_stamp'].values[0]
    exp_end = stim_df.loc[stim_df.event == 'Offset_'+task, 'lsl_time_stamp'].values[0]
    exp_dur = round(exp_end - exp_start, 4)

    # expected mm:ss
    exp_dt = datetime.timedelta(seconds=exp_dur)
    exp_dt_dur = str(datetime.timedelta(seconds=round(exp_dt.total_seconds())))

    for stream in streams:
        # don't include mic in resting state
        if task == 'RestingState' and stream == 'mic':
            continue
        
        duration_name = f'{stream}_duration'
        mmss_name = f'{stream}_mm:ss'
        percent_name = f'{stream}_percent'

        if error_map[stream]: 
            subject = xdf_path.split('sub-')[-1].split('_')[0]#xdf_path.split('/')[6].split('-')[1]
            print(f'No {stream} data for participant {subject}')
            continue
        # grab data for stream + experiment part
        event_data = get_event_data(task, df_map[stream], stim_df)

        # print if no data
        if event_data.empty:
            durations_dict[duration_name] = 0
            durations_dict[mmss_name] = str(datetime.timedelta(seconds=0))
            durations_dict[percent_name] = '0.0000'
            print(f'{stream} has no {task} data') 
            continue
        # calculate duration
        start = event_data['lsl_time_stamp'].values[0]
        stop = event_data['lsl_time_stamp'].values[-1]
        dur = round(stop - start, 4)

        # calculate hh:mm:ss
        dt = datetime.timedelta(seconds=dur)
        dt_dur = str(datetime.timedelta(seconds=round(dt.total_seconds())))

        # calculate percent 
        percent = round(dur/exp_dur * 100, 4)

        durations_dict[duration_name] = dur
        durations_dict[mmss_name] = dt_dur
        durations_dict[percent_name] = percent
        if dur == 0: ## do i need this?
            continue
        if dur < (exp_dur - 5): # 5 second margin
            print(f"{stream} is shorter than expected for {task} by {exp_dur - dur:.4f} seconds")
    durations_dict[duration_name] = exp_dur
    durations_dict[mmss_name] = exp_dt_dur
    durations_dict[percent_name] = '100.0'

    return durations_dict


def whole_durations(xdf_path: str, stim_df: pd.DataFrame, df_map: dict, error_map: dict) -> pd.DataFrame:
    """
    Get the durations of each data stream and compare to their expected duration, for the entire experiment, where the expected duration is 
    the max duration of any data stream.
    Args:
        xdf_path (str): The path to the xdf file.
        stim_df (pd.DataFrame): The stimuli dataframe containing the events mapped to lsl timestamps.
        df_map (dict): Contains dataframes for each data modality, loaded through import_modality_data functions in utils.
        error_map (dict): Contains booleans for each data modality indicating error. 

    Returns:
        pd.DataFrame: The durations of each stream in seconds and mm:ss and the percent that that duration comprised 
        of the max duration of all data streams. 
    """
    if error_map['behavior']:
        return None

    streams = list(df_map.keys())

    whole_durations_df = pd.DataFrame(columns = ['stream', 'duration', 'mm:ss'])
  
    # populate whole_durations_df
    for i, stream in enumerate(streams): 
        if error_map[stream]:
            subject = xdf_path.split('/')[6].split('-')[1]
            print(f'No {stream} data for participant {subject}')
            duration = float('nan')
            whole_dt_dur = ''
            whole_durations_df.loc[i] = [stream, duration, whole_dt_dur]
            continue
        duration = df_map[stream]['lsl_time_stamp'].iloc[-1]- df_map[stream]['lsl_time_stamp'].iloc[0]
        duration = round(duration, 4)
        # convert to mm:ss
        whole_dt = datetime.timedelta(seconds=duration)
        whole_dt_dur = str(datetime.timedelta(seconds=round(whole_dt.total_seconds())))
        whole_durations_df.loc[i] = [stream, duration, whole_dt_dur]
    
    whole_durations_df.sort_values(by = 'duration', inplace = True)

    # percent
    max_dur = whole_durations_df.duration.max()
    whole_durations_df['percent'] = whole_durations_df['duration'].apply(lambda x: round(x / max_dur * 100, 4) )

    # print which are short
    for i in whole_durations_df.iterrows():
        if i[1]['duration'] == 0:
            continue
        if i[1]['duration'] < (max_dur - 30): # 30 second margin
            print(f"{i[1]['stream']} is shorter than expected by {max_dur - i[1]['duration']:.4f} seconds")

        
    whole_durations_df.sort_values(by = 'duration', inplace = True)
    return whole_durations_df