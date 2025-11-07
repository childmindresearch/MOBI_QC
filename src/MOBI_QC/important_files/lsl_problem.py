import pyxdf
import pandas as pd
import numpy as np
from glob import glob
import datetime
import matplotlib.pyplot as plt
from pprint import pprint
from utils import *
import math

def lsl_quick_check(ps_df: pd.DataFrame):
    """
    Quick check for gaps in LSL timestamps using physio data.
    Args:
        ps_df (pd.DataFrame): Dataframe containing the physio data.
    Returns:
        quickcheck (int): Number of instances where the difference between consecutive LSL timestamps is not close to inverse of sampling rate.
    """
    sampling_rate = get_sampling_rate(ps_df)
    quickcheck = sum([not math.isclose(x, 1/sampling_rate, abs_tol=1e-2) for x in ps_df.lsl_time_stamp.diff()]) - 1
    return quickcheck

    
def lsl_loss_percentage(df_map: dict, error_map: dict, sub_id: str) -> dict:
    """
    Calculate the percentage of data loss for each modality based on LSL timestamps.
    Args:
        df_map (dict): Dictionary containing dataframes for each modality.
        error_map (dict): Contains booleans for each data modality indicating error. 
        sub_id (str): Subject ID.
    Returns:
        percent_data_loss (pd.DataFrame): Dataframe containing the percentage of data loss for each modality.
    """
    # df with percent loss (diff greater than median)
    modalities = list(df_map.keys())
    lsl_dict = {}

    for modality in modalities:
        num_losses_name = f'{modality}_num_losses'
        percent_name = f'{modality}_percent_lost'

        if error_map[modality]: 
            print(f'No {modality} data for participant {sub_id}')
            loss_instances = float('nan')
            percent_lost = float('nan')
            lsl_dict[num_losses_name] = loss_instances
            lsl_dict[percent_name] = percent_lost
            continue
        df = df_map[modality]

        # median diff between lsl_time_stamp (with 1.05 margin) 
        df['diff'] = df['lsl_time_stamp'].diff()
        median = df['diff'].median() * 1.05
        
        # number of loss instances  
        loss_instances = (df['diff'] > median).sum()
        if loss_instances != 0:
            # amount of data skipped: values for which diff>median 
            amt_data_lost = df.loc[df['diff'] > median, 'diff'].values[0].sum()
            # total amount of data: last - first lsl_time_stamp
            amt_data_total = df['lsl_time_stamp'].values[-1] - df['lsl_time_stamp'].values[0]
            
            percent_lost = amt_data_lost/amt_data_total * 100
        else:
            percent_lost = 0
    
        lsl_dict[num_losses_name] = loss_instances
        lsl_dict[percent_name] = round(percent_lost, 4)

    return lsl_dict
    
def lsl_loss_before_social(df_map: dict, error_map: dict, sub_id: str, offset_social_timestamp: float) -> pd.DataFrame:
    """
    Calculate the percentage of data loss before the social task offset for each modality.
    Args:
        df_map (dict): Dictionary containing dataframes for each modality.
        error_map (dict): Contains booleans for each data modality indicating error. 
        sub_id (str): Subject ID.
        offset_social_timestamp (float): Timestamp of  social task offset.
    Returns:
        lsl_social_dict (dict): Dictionary containing the number and percentage of data loss before the social task offset for each modality.
    """

    modalities = list(df_map.keys())
    lsl_social_dict = {}

    for modality in modalities:
        num_losses_name = f'{modality}_num_losses'
        percent_name = f'{modality}_percent_lost'

        if error_map[modality]: 
            print(f'No {modality} data for participant {sub_id}')
            loss_instances = float('nan')
            percent_lost = float('nan')
            lsl_social_dict[num_losses_name] = loss_instances
            lsl_social_dict[percent_name] = percent_lost
            continue
        df = df_map[modality]
        df['diff'] = df['lsl_time_stamp'].diff()
        social_df = df.loc[df.lsl_time_stamp <= offset_social_timestamp]

        # median diff between lsl_time_stamp (with 1.05 margin) 
        median = df['diff'].median() * 1.05

        # number of loss instances  
        loss_instances = (social_df['diff'] > median).sum()
        percent_lost = 0
        amt_data_lost = 0

        # LSL loss starts and ends before offset_social
        if loss_instances != 0:
            # amount of data skipped: values for which diff>median 
            amt_data_lost = social_df.loc[social_df['diff'] > median, 'diff'].values[0].sum()

        # offset social is between LSL loss onset + offset
        remaining_lost = offset_social_timestamp - social_df['lsl_time_stamp'].values[-1]
        if (remaining_lost) > 1:
            loss_instances +=1
            amt_data_lost = amt_data_lost + remaining_lost

        amt_data_total = offset_social_timestamp - social_df['lsl_time_stamp'].values[0]
        percent_lost = amt_data_lost/amt_data_total * 100


        lsl_social_dict[num_losses_name] = loss_instances
        lsl_social_dict[percent_name] = round(percent_lost, 4)  

    return lsl_social_dict

def lsl_problem_qc(xdf_filename:str, stim_df:pd.DataFrame, df_map:dict, error_map:dict) -> dict:
    """
    Main function to check for LSL timestamp gaps in the data.
    Args:
        xdf_filename (str): Path to the XDF file.
        stim_df (pd.DataFrame): Contains stimulus markers
        df_map (dict): Dictionary with all data dfs 
        error_map (dict): Contains booleans for each data modality indicating error. 

        Returns:
        vars (dict): Dictionary containing the percentage of data loss for each modality and the number of loss instances.
    """
    # load data 
    sub_id = xdf_filename.split('sub-')[1].split('/')[0]

    # behavior error handling 
    if error_map['behavior']:
        lsl_behavior_error_dict = {}
        modalities = list(df_map.keys())
        for modality in modalities:
            num_losses_name = f'{modality}_num_losses'
            percent_name = f'{modality}_percent_lost'
            lsl_behavior_error_dict[num_losses_name] = float('nan')
            lsl_behavior_error_dict[percent_name] = float('nan')
        return lsl_behavior_error_dict

    offset_social_timestamp = stim_df.loc[stim_df['event'] == 'Offset_SocialTask', 'lsl_time_stamp'].values[0]

    # optional: returns number of loss instances in ps_df
    # lsl_quick_check(ps_df)
    # ps_df = df_map['ps']

    # not plotting anymore
    # plot_df = df_map[modality_to_plot]
    # lsl_problem_plot(plot_df, sub_id, modality_to_plot)

    # loss_in_experiment_vars = lsl_loss_percentage(df_map, error_map, sub_id)

    loss_before_social_vars = lsl_loss_before_social(df_map, error_map, sub_id, offset_social_timestamp)

    return loss_before_social_vars

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass