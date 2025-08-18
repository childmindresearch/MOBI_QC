from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.utils import ImageReader
import os
import mne
import json
import glob

from utils import *
from eeg_qc import compute_eeg_pipeline, test_eeg_pipeline
from ecg_qc import ecg_qc 
from eda_qc import eda_qc
from rsp_qc import *
from mic_qc import *
from lsl_problem import *
from et_qc import *
from webcam_qc import *
from behavior_qc import *
from generate_csv import *

import seaborn as sns
import matplotlib.pyplot as plt

def run_qc(xdf_filename):
    stim_df = import_stim_data(xdf_filename)
    error_log = []
    subjectID = subject.split('-')[1].split('/')[0]

    eeg_vars, eeg_df, eeg_error = compute_eeg_pipeline(xdf_filename, 
                                                            stim_df=stim_df, 
                                                            task='RestingState')

    [ecg_vars, ecg_plt, ps_df, ecg_error] = ecg_qc(xdf_filename = xdf_filename, stim_df = stim_df, task='RestingState')

    [eda_vars, eda_plt1, eda_plt2, ps_df, eda_error] = eda_qc(xdf_filename = xdf_filename, stim_df = stim_df, task= 'RestingState')

    rsp_vars, ps_df, rsp_error = rsp_qc(xdf_filename=xdf_filename, stim_df=stim_df, task='RestingState')

    mic_vars, mic_df, mic_error = mic_qc(xdf_filename=xdf_filename, stim_df=stim_df)

    video_filename = '/'.join(xdf_filename.split('/')[:-1])+ f'/sub-{subjectID}_task-CUNY_run-001_video.avi'
    webcam_vars, cam_df, cam_error = webcam_qc(xdf_filename=xdf_filename,
                                                video_file=video_filename, 
                                                stim_df=stim_df,task='RestingState')

    et_vars, et_df, et_error = et_qc(xdf_filename = xdf_filename, stim_df = stim_df, task='RestingState')

    behavior_vars, behavior_error = behavior_qc(xdf_filename, stim_df)

    df_map = {
        'et': et_df,
        'ps': ps_df,
        'mic': mic_df,
        'cam': cam_df,
        'eeg': eeg_df
        }

    cam_error = False # remove this after webcam done
    ps_error = False
    error_map = {
        'et': et_error,
        'ps': ps_error,
        'mic': mic_error,
        'cam': cam_error,
        'eeg': eeg_error
        }

    duration_vars = {"Durations of each modality + comparison to expected duration:": 
        get_durations(xdf_path=xdf_filename, task='Experiment', stim_df = stim_df, df_map = df_map, error_map = error_map)}

    lsl_vars = lsl_problem_qc(xdf_filename, stim_df=stim_df, df_map=df_map, error_map = error_map)

    modality_vars = {
    'eeg':eeg_vars, 
    'et':et_vars, 
    'ecg':ecg_vars, 
    'eda':eda_vars, 
    'rsp':rsp_vars, 
    'mic':mic_vars, 
    'cam':webcam_vars,
    'behavior': behavior_vars, 
    'lsl':lsl_vars, 
    'duration':duration_vars}

    return modality_vars


subject = "P5402677"
xdf_filename =  f'/Users/apurva.gokhe/Documents/CUNY_QC/data/sub-{subject}/sub-{subject}_ses-S001_task-CUNY_run-001_MOBI.xdf'

video_filename = '/'.join(xdf_filename.split('/')[:-1])+ f'/sub-{subject}_task-CUNY_run-001_video.avi'

stim_df = import_stim_data(xdf_filename)

error_log = []

modaity_vars = run_qc(xdf_filename)
generate_csv(xdf_filename, modality_vars)

# Report code goes here