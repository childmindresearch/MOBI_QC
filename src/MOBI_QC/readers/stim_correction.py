#%%
import pyxdf    # 1.17.0'
import pandas as pd    # 2.2.3
import numpy as np    #  2.26
from glob import glob
import sys
import os
from scipy.io import wavfile    #1.11.3
from scipy.signal import correlate, resample    #1.11.3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))






def trigger_recovery(stim_df, xdf_filename):
    ''' This function recovers the trigger events in the stim_df DataFrame'''   
    # Import the Mic data
    mic_data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'Microphone'}], verbose = False)
    mic_df = pd.DataFrame(mic_data[0]['time_series'], columns=['int_array'])
    mic_df['time_stamp'] = mic_data[0]['time_stamps']
    # Reduce the mic_data to only the story listening events, we have triggers for onset and offset of the story listening block
    mic_df = mic_df.loc[(mic_df.time_stamp >= stim_df.loc[stim_df.event == 'Onset_StoryListening', 'time_stamp'].values[0]) & 
                    (mic_df.time_stamp <= stim_df.loc[stim_df.event == 'Offset_StoryListening', 'time_stamp'].values[0])]
    mic = mic_df['int_array'].values

    # Import the behavior file from psychopy
    psy_path = glob(os.path.join('/'.join(xdf_filename.split('/')[:-1]),'*behavior.csv'))[0]
    psycho = pd.read_csv(psy_path, sep=',', header=0)
    audio_order = [f.split('/')[-1].split('.')[0] for f in psycho.AudioFile.unique() if pd.notna(f)]
    # Add triggers
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

    story_triggers = []
    for story in audio_order:
        # import the audio file and downsample to 44.1kHz
        audio_file_path = 'NEW_AUDIO_48/'+ story + '.wav'
        fs_audiofile, audiofile = wavfile.read(audio_file_path)
        audio_duration = len(audiofile) / fs_audiofile
        audiofile = resample(audiofile, int(audio_duration * 44100))

        # cross correlate with the microphone data
        corr = correlate(mic, audiofile, mode='full')
        best_index = np.argmax(corr)
        onset = best_index - len(audiofile) + 1
        onset_timestamp = mic_df.time_stamp[onset]
        story_triggers.append(onset_timestamp)

        # calculate the offset from the length of the audio file
        offset = onset + len(audiofile)
        offset_timestamp = mic_df.time_stamp[offset]
        story_triggers.append(offset_timestamp)

        # add to stim_df
        if story == 'Camp_Lose_A_Friend':
                event_id = 20
        elif story == 'Frog_Dissection_Disaster':
            event_id = 30
        elif story == 'I_Decided_To_Be_Myself_And_Won_A_Dance_Contest':
            event_id = 40
        elif story == 'I_Fully_Embarrassed_Myself_In_Zoom_Class1':
            event_id = 50
        elif story == 'Left_Home_Alone_in_a_Tornado':
            event_id = 60
        elif story == 'The_Birthday_Party_Prank':
            event_id = 70

        # Add the story onset to the stim_df
        stim_df.loc[len(stim_df)] = [event_id, onset_timestamp, f'{events[event_id]}']
        # Add the story offset to the stim_df
        stim_df.loc[len(stim_df)] = [event_id + 1, offset_timestamp, f'{events[event_id+1]}', ]


    #sort by the time_stamp
    stim_df.sort_values('time_stamp', inplace=True)

    return stim_df


# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass