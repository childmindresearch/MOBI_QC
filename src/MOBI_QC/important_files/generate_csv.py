import os
import glob
import pandas as pd
from utils import *
from eeg_qc import *
from et_qc import *
from ecg_qc import *
from eda_qc import *
from rsp_qc import *
from mic_qc import *
from webcam_qc import *
from behavior_qc import *
from lsl_problem import *

def generate_qc_dataframe(csvfilename, subject_id:str, collection_date: str, modality_vars_values):
    if os.path.exists(csvfilename) == True:
        #csv_info = pd.read_csv(csvfilename)
        qc_columns = (pd.read_csv(csvfilename)).columns
        subject_csv = {'Subject': subject_id, 'Collection Date': collection_date}
        for metric_pair in modality_vars_values:
            subject_csv.update(metric_pair)
        if list(qc_columns) != list(subject_csv.keys()):
            subject_csv = {key: subject_csv[key] for key in qc_columns}
    else:
        subject_csv = {'Subject': subject_id, 'Collection Date': collection_date}
        for metric_pair in modality_vars_values:
            subject_csv.update(metric_pair)
    subject_csv_df = pd.DataFrame([subject_csv])

    return subject_csv_df

def check_data_exists(filename, subject_id):
    existing_data = pd.read_csv(filename)  
    if subject_id in existing_data['Subject'].values:
        rows = existing_data.loc[existing_data['Subject'] == subject_id]
        return rows
    else:
        rows = pd.DataFrame()
        return rows

def save_to_csv(subject_csv_df:pd.DataFrame):
    if os.path.exists('CUNY_QC.csv') == True:
        subject_csv_df.to_csv('CUNY_QC.csv', mode='a', index=False, header=False)
        return 'Saved'
    elif os.path.exists('CUNY_QC.csv') == False:
        subject_csv_df.to_csv('CUNY_QC.csv', index=False, header=True)
        return 'Saved'
    else:
        return 'Export Error'
"""
def add_to_csv(subject_id, collection_date, modality_vars):
    modality_vars['lsl_vars'], modality_vars['duration_vars'] = unpack_vars(modality_vars['lsl_vars'], modality_vars['duration_vars'])
    subject_csv_df = generate_qc_dataframe(subject_id, collection_date, modality_vars.values())
    added_to_csv = save_to_csv(subject_csv_df)
    return added_to_csv
"""
def generate_csv(xdf_filename:str, modality_vars:dict):

    csvfilename = 'CUNY_QC.csv'
    subject_id = xdf_filename.split('sub-')[1].split('/')[0]
    collection_date = get_collection_date(xdf_filename)
    if os.path.exists(csvfilename) == True and (check_data_exists(csvfilename, subject_id)).empty != True:
        print ('Participant data already exists')
        return check_data_exists(csvfilename, subject_id)
    else:
        subject_csv_df = generate_qc_dataframe(csvfilename, subject_id, collection_date, modality_vars.values())
        status = save_to_csv(subject_csv_df)
        #status = add_to_csv(subject_id, collection_date, modality_vars)
        return status


# %%

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass

# %%
