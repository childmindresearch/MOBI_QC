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

def get_subjectID_and_collectionDate(xdf_filename:str):
    
    subject_id = xdf_filename.split('sub-')[1].split('/')[0]
    collection_date = get_collection_date(xdf_filename)
    
    return subject_id, collection_date

def unpack_vars(lsl_vars, duration_vars):
    # Unpack lsl vars
    lsl_prob_vars = {}
    stream_vars = {}

    for key in lsl_vars.keys():
        if isinstance(lsl_vars[key], pd.DataFrame):
            lsl_k = lsl_vars[key].to_dict(orient='records')
            for i in range(len(lsl_k)):
                lsl_dict = {f"{lsl_k[i]['stream']}_{k}": v for k, v in lsl_k[i].items() if k != "stream" and k!='subject'}
                lsl_prob_vars.update(lsl_dict)
        else:
            lsl_prob_vars.update({key:lsl_vars[key]})
    
    # Unpack stream durations
    stream = duration_vars['Durations of each modality + comparison to expected duration:'].to_dict(orient='records')
    stream_vars = {}
    for i in range(len(stream)):
        stream_dict = {f"{stream[i]['stream']}_{k}": v for k, v in stream[i].items() if k != "stream"}
        stream_vars.update(stream_dict)
    return lsl_prob_vars, stream_vars

def add_modality_name(modality_vars):
    for key, vars in modality_vars.items():
        new_vars = {(f"{key}_{k}" if key not in k else k): value for k, value in vars.items()}
        modality_vars[key] = new_vars
    return modality_vars

def generate_qc_dataframe(subject_id:str, collection_date: str, modality_vars):
    
    subject_csv = {'Subject': subject_id, 'Collection Date': collection_date}
    for modality in modality_vars:
        subject_csv.update(modality)
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
    
def make_iqr_plots(subject_id, modality_vars):
    if os.path.exists('CUNY_QC.csv') == False:
        return 'QC metric csv file does not exist.'
    else:
        qc_metrics = pd.read_csv('CUNY_QC.csv')
        print(qc_metrics['Subject'].nunique())
        if qc_metrics['Subject'].nunique() <= 10:
            return 'IQR cannot be calculated due to insufficient participant data.'
        else:
            numeric_cols = qc_metrics.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                plt.figure(figsize=(5, 2.5))
                sns.boxplot(x=qc_metrics[col], color='paleturquoise', width=0.2)
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for modality in modality_vars:
                    if col in modality_vars[modality]:
                        subject_value = modality_vars[modality][col]
                        print(col,'=',subject_value)
                        plt.axvline(subject_value, color='red', linestyle='dotted', linewidth=1, label=f'N={qc_metrics.shape[0]}')
                        xlim = plt.gca().get_xlim()
                        if float(subject_value) < (xlim[0]+xlim[1])/2:
                            text_x = xlim[1]
                            side = 'right'
                        else:
                            text_x = xlim[0]
                            side='left'
                        #plt.vlines(subject_value, ymin=qc_metrics[col].min(), ymax=qc_metrics[col].max(), color='red', zorder=5, label=subject)
                        break
                plt.text(
                    text_x, plt.gca().get_ylim()[0],
                    f'N={qc_metrics.shape[0]}',
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment=side
                )
                plt.xlabel(None)
                plt.ylabel(col)
                plt.tight_layout()
                plt.savefig(f'report_images/IQR_imgs/{subject_id}_{col}_IQR.png')
                plt.show()
            return 'IQR plots were created.'
    
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
    subject_id, collection_date = get_subjectID_and_collectionDate(xdf_filename)
    if os.path.exists(csvfilename) == True and (check_data_exists(csvfilename, subject_id)).empty != True:
        print ('Participant data already exists')
        return check_data_exists(csvfilename, subject_id)
    else:
        modality_vars['lsl'], modality_vars['duration'] = unpack_vars(modality_vars['lsl'], modality_vars['duration'])
        modality_vars = add_modality_name(modality_vars)
        create_IQR = make_iqr_plots(subject_id, modality_vars)
        subject_csv_df = generate_qc_dataframe(subject_id, collection_date, modality_vars.values())
        status = save_to_csv(subject_csv_df)
        #status = add_to_csv(subject_id, collection_date, modality_vars)
        return create_IQR, status


# %%

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass

# %%
