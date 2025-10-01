from glob import glob
import pandas as pd
import wave

from MOBI_QC.readers.readers import BaseProcessor


def check_audio_file(b: BaseProcessor) -> pd.DataFrame:
    """    
    Checks the audio file associated with the subject data.    
    Parameters:
    -----------
    subdat : BaseProcessor
        An instance of the BaseProcessor class containing subject data.
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the results of the audio file check.
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
    """
    This function checks for missing markers.
    Args:
        events (dict[int, str]): The Dictionary that includes all stimulus markers adn corresponding labels
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps    
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
    

def get_seconds_between_triggers(b: BaseProcessor, trigger1: int, trigger2: int) -> float:
    """
    This function computes the duration between two event triggers in seconds.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps 
        trigger1 (int): The end trigger for duration calculation
        trigger2 (int): The start trigger for duration calculation
    Returns:
        duration_between_triggers (float): The duration between given triggers in seconds  
    """   

    t0 = b.stim.data.loc[b.stim.data.trigger == trigger1, 'time_stamp'].values[0]
    t1 =  b.stim.data.loc[b.stim.data.trigger == trigger2, 'time_stamp'].values[0]

    return abs(t1 - t0)


def total_experiment_duration(b: BaseProcessor) -> str:

    """
    This function computes the total duration of the experiment.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps    
    Returns:
        total_duration (str): The total duration of the experiment in minutes:seconds   
    """    
    minutes_entire_experiment, seconds = divmod(get_seconds_between_triggers(b, 201, 200), 60) 
    #total_duration = f"{int(minutes_entire_experiment):02}:{int(seconds):02}"
    total_duration_in_seconds = float(minutes_entire_experiment*60) + float(seconds)

    return total_duration_in_seconds

def unexpected_durations(b:BaseProcessor, audiofiles: list) -> list | None:
    """
    This function checks whether story listening task durations are of expected length.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps.
        story_onsets (list): The list of story listening event triggers.
        events (dict): The dictionary that contains all event triggeres and their labels.
        audiofiles (list): The list of paths to all story listening audiofiles.    
    Returns:
        list_of_task_duration_difference (list): The story listening tasks with durations not within expected length. 
    """    
    story_onsets = [20,30,40,50,60,70]
    durations = pd.DataFrame({
    'trigger':story_onsets,
    'story':[b.events[x] for x in story_onsets],
    'lsl_duration': [get_seconds_between_triggers(b, x+1, x) for x in story_onsets],
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