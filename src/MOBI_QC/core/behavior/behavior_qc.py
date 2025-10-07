import wave  # noqa: D100
from glob import glob

import numpy as np
import pandas as pd

from MOBI_QC.readers.readers import BaseProcessor


def check_audio_file(b: BaseProcessor) -> pd.DataFrame:
    """Checks the audio file associated with the subject data
    
    Args:
        b : An instance of the BaseProcessor class containing subject data. 
    Returns:
        pd.DataFrame: A DataFrame containing the results of the audio file check.
    """
    
    behavior_filepath = glob('/'.join(b.xdf_path.split('/')[:-1])+'/*_behavior.csv')[0]
    behavior_csv = pd.read_csv(behavior_filepath, keep_default_na=False)
    audio_files = behavior_csv['AudioFile'].unique()
    audiofile_df = pd.DataFrame(columns = ['Story', 'SamplingFreq'])
    for audio in audio_files:
        if audio != '':
            story = audio.split('/')[-1]
            #audiofreq = audio.split('normalized_')[1].split('/')
            if '48k' in (str)(audio):
                audiofreq = '48k'
            else:
                audiofreq = '44k'
            audiofile_df.loc[len(audiofile_df)] = {'Story': story, 'SamplingFreq': audiofreq}

    return audiofile_df

def get_missing_markers(b: BaseProcessor) -> list | None:
    """This function checks for missing markers.

    Args:
        b : An instance of the BaseProcessor class containing subject data.
    
    Returns:
        missing_markers (list): The list of missing event markers in the given xdf file   
    """    
    missing_markers=[]
    for event in list(b.events.values()):
        if event in b.stim.data.event.tolist():
            continue
        else:
            missing_markers = missing_markers + [event]
    if not missing_markers:
        return ['None'] # this (as opposed to NoneType) helps with report creation and readability
    else:   
        return missing_markers
    
def event_duration(b: BaseProcessor, onset_label: str, offset_label: str, in_seconds: bool=False) -> float | tuple:
    """This function computes the duration between two event triggers in seconds.

    Args:
        b (BaseProcessor): An instance of the BaseProcessor class containing subject data
        onset_label (str): The onset label for duration calculation
        offset_label (str): The offset label for duration calculation
        in_seconds (bool): If True, duration is returned in seconds. If False, duration is returned in minutes and seconds.

    Returns:
        duration_between_triggers (float | tuple): The duration between given triggers in seconds or (minutes, seconds)
    """
    if in_seconds:
        seconds = (s := b.stim.subset(b.stim.data, onset_label, offset_label)["time_stamp"]).iloc[-1] - s.iloc[0]
        #print("{:.4f} sec".format(seconds))
        return seconds
        
    else:
        min, sec = divmod((
        s := b.stim.subset(b.stim.data, onset_label, offset_label)["time_stamp"]).iloc[-1] - s.iloc[0], 60)
        #print("{:.0f} min {:.2f} sec".format(*[min, sec]))
        return min, sec
    
def unexpected_durations(b:BaseProcessor, audiofiles: list) -> list | None:
    """This function checks whether story listening task durations are of expected length.

    Args:
        b : An instance of the BaseProcessor class containing subject data.
        audiofiles (list): The list of paths to all story listening audiofiles.
            
    Returns:
        list_of_task_duration_difference (list): The story listening tasks with durations not within expected length. 
    """    
    story_onsets = [20,30,40,50,60,70]
    durations = pd.DataFrame({
    'trigger':story_onsets,
    'story':[b.events[x] for x in story_onsets],
    'lsl_duration': [event_duration(b, b.events[x], b.events[x+1], in_seconds=True) for x in story_onsets],
    'audiofile_duration': [wave.open(x).getnframes()/wave.open(x).getframerate() for x in audiofiles], #duration of audio file is number of frames divided by the frame rate.
    'audio_sampling_freq': [x.split('NEW_AUDIO_')[-1].split('/')[0] for x in audiofiles]
    })

    durations['difference(sec)'] = durations['audiofile_duration'] - durations['lsl_duration']
    
    task_duration_difference = []

    # Calculating audiofile duration in 48kHz and then comparing with story listening durations from stim_df
    for i in range(len(durations.audiofile_duration)):
        if durations.audio_sampling_freq[i] == '44k':
            task_duration = (durations.audiofile_duration[i] * 44100) / 48000
        else:
            task_duration = durations.audiofile_duration[i]
        if (durations.lsl_duration[i].round(3) -  task_duration.round(3)) > 0.5:
            task_duration_difference = task_duration_difference + [durations.story[i]]

    if task_duration_difference != []:
        list_task_duration_difference = task_duration_difference
    else:
        list_task_duration_difference = None

    return list_task_duration_difference

    """This function checks whether all 10-second rest periods are approximately 10 seconds long.

    Args:
        b (BaseProcessor): An instance of the BaseProcessor class containing subject data

    Returns:
        bool: True if all 10-second rest periods are approximately 10 seconds long, False otherwise
    """
    rests = b.stim.data.loc[(b.stim.data['trigger'] == 100)| (b.stim.data['trigger'] == 101)]
    rest_durations = [rests.iloc[i+1]['time_stamp'] - rests.iloc[i]['time_stamp'] for i in range(0, len(rests), 2)]
    return all(9.9 <= duration <= 10.1 for duration in rest_durations)

def ten_seconds_rest(b: BaseProcessor) -> bool:
    """This function checks whether all 10-second rest periods are approximately 10 seconds long.

    Args:
        b (BaseProcessor): An instance of the BaseProcessor class containing subject data

    Returns:
        bool: True if all 10-second rest periods are approximately 10 seconds long, False otherwise
    """
    rests = b.stim.data.loc[(b.stim.data['trigger'] == 100)| (b.stim.data['trigger'] == 101)]
    rest_durations = [rests.iloc[i+1]['time_stamp'] - rests.iloc[i]['time_stamp'] for i in range(0, len(rests), 2)]
    # check if all rest durations are approximately 10 seconds (with a tolerance of .1 second)
    return all(9.9 <= duration <= 10.1 for duration in rest_durations)

def calculate_average_response_time(b:BaseProcessor) -> float:
    """This function calculates the average response time for subject input events.

    Args:
        b (BaseProcessor): An instance of the BaseProcessor class containing subject data
    Returns:
        float: The average response time in seconds
    """
    resps = b.stim.data.loc[(b.stim.data['trigger'] == 300)| (b.stim.data['trigger'] == 301)]
    resp_durations = [resps.iloc[i+1]['time_stamp'] - resps.iloc[i]['time_stamp'] for i in range(0, len(resps), 2)]
    return np.mean(resp_durations)

def behavior_qc(b:BaseProcessor) -> float:
    """This function processes behavioral data stream to compute quality metrics.

    Args:
        b (BaseProcessor): An instance of the BaseProcessor class containing subject data
    Returns:
        bx_vars (dict): A dictionary containing computed behavioral quality metrics
        behavior_error (bool): True if any errors were encountered during processing, False otherwise
    """
    audiofiles_df = check_audio_file(b)
    audiofiles=[]
    for story in audiofiles_df['Story']:
        freq = audiofiles_df.loc[audiofiles_df['Story'] == story, 'SamplingFreq'].iloc[0]
        audiofiles = audiofiles + [f"NEW_AUDIO_{freq}/{story}"]

    bx_vars = {}
    try:
        bx_vars['missing_stimulus_markers'] = get_missing_markers(b)
        print(f"Missing markers: {bx_vars['missing_stimulus_markers']}")
        bx_vars['total_duration'] = event_duration(b, "Onset_Experiment", "Offset_Experiment", in_seconds=False)
        print(f"Total duration (min, sec): {bx_vars['total_duration']}")
        bx_vars['unexpected_durations'] = unexpected_durations(b, audiofiles)
        print(f"Unexpected story durations: {bx_vars['unexpected_durations']}")
        bx_vars['impedance_check_duration'] = event_duration(b, "Onset_impedanceCheck", "Offset_impedanceCheck", in_seconds=False)
        print(f"Impedance check duration (min, sec): {bx_vars['impedance_check_duration']}")
        bx_vars['ten_second_rest'] = ten_seconds_rest(b)
        print(f"All 10-second rest periods approximately 10 seconds long: {bx_vars['ten_second_rest']}")
        bx_vars['average_response_time'] = calculate_average_response_time(b)
        print(f"Average response time (sec): {bx_vars['average_response_time']}")
        behavior_error = False
        return bx_vars, behavior_error
    except IndexError:
        print(f'Error: Missing stimulus markers for participant {b.subject} in {b.xdf_path}.')
        bx_vars.update({key:float('nan') for key in bx_vars.keys()})
        bx_vars['missing_stimulus_markers'] = get_missing_markers(b)
        behavior_error = True
        return bx_vars, behavior_error


if __name__ == "__main__":
    pass