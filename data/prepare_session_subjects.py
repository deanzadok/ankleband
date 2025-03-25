import os
import sys
sys.path.append('.')
import time
import glob
import argparse
import datetime
import pandas as pd
import numpy as np
from data.prepare_session import prepare_session_data

def prepare_session_for_subjects(data_dir, output_dir, req_subjects=None):

    # initiate stats dataframe
    stat_df = pd.DataFrame()

    # iterate over all subjects folders
    for subject_dir in glob.glob(os.path.join(args.data_dir, "ID*/")):

        subject_name = subject_dir.split('/')[-2]

        if req_subjects is None or int(subject_name.split('ID')[-1]) in req_subjects:

            # iterate over all relevant sessions folders in the subject's folder
            for session_dir in glob.glob(os.path.join(subject_dir, "*/")):

                session_name = session_dir.split('/')[-2]

                # interpolate vicon data
                print(f'Preparing {subject_name}:{session_name}...')

                stat_dict = prepare_session_data(data_dir=session_dir, output_dir=output_dir, verbose=False)
                stat_dict['subject'] = subject_name
                stat_dict['session'] = session_name

                # add stats of session file
                stat_df = pd.concat([stat_df, pd.DataFrame([stat_dict])], ignore_index=True)
            
    # reorder columns and save
    first_columns = ['subject', 'session']
    stat_df = stat_df[first_columns + [x for x in stat_df.columns.values if x not in first_columns]]
    csv_file_name = 'h5_prep_stats.csv'
    stat_df.to_csv(os.path.join(data_dir, csv_file_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path to folder containing data files', default='/Data/imugr/datasets/bracelet', type=str)
    parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='/Data/imugr/datasets/bracelet/prepared_data', type=str)
    args = parser.parse_args()

    # req_subjects = [1, 2, 4, 6, 8]
    # prepare_session_for_subjects(data_dir=args.data_dir, output_dir=args.output_dir, req_subjects=req_subjects)
    prepare_session_for_subjects(data_dir=args.data_dir, output_dir=args.output_dir)