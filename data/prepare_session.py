import os
import sys
sys.path.append('.')
import time
import glob
import argparse
import datetime
import pandas as pd
import numpy as np
from data.interpolate_vicon import convert_time_to_datetime, separate_link_columns, convert_vicon_data_to_numeric

def prepare_session_data(data_dir, output_dir, num_classes=4, verbose=True):
    
    # load old vicon csv to mark time intervals without labelling data
    stat_dict = {}
    vicon_df, stat_dict = load_vicon_csv(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

    # load labels
    labels_df, stat_dict = load_labels_data(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

    # compute gestures percentage
    labels_count = labels_df.label.value_counts().to_dict()
    del labels_count[0]
    gestures_percentage = np.array([val for key, val in labels_count.items()]).sum() / len(labels_df)
    if verbose:
        print('Gestures Percentage: {}'.format(gestures_percentage))
    stat_dict['gestures_percentage'] = gestures_percentage

    # detect missing intervals and remove from data
    labels_df, stat_dict = detect_missing_vicon_intervals(vicon_df=vicon_df, stat_dict=stat_dict, labels_df=labels_df, verbose=verbose)

    # load imu data
    imu_df, stat_dict = load_imu_data(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

    # merge all data into one dataframe and sort according to datetime
    if verbose:
        print('Merging IMU measurements with Vicon data...')
    data_df = pd.concat([labels_df, imu_df])
    data_df.sort_values(by='time', inplace=True)
    data_df['label'] = data_df['label'].ffill().bfill()
    data_df.dropna(inplace=True)

    if len(data_df) < len(imu_df) and verbose:
        print(f'Lost {len(imu_df) - len(data_df)} samples when merging IMU data with Vicon data')
    stat_dict['lost_in_merge'] = len(imu_df) - len(data_df)

    # check if output folder exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # save as h5
    save_processed_h5(data_df=data_df, output_dir=output_dir, data_dir=data_dir, verbose=verbose)

    return stat_dict

# load vicon csv as dataframe
def load_vicon_csv(data_dir, stat_dict, verbose=True):

    # create csv path
    vicon_dir = os.path.join(data_dir, "vicon")
    csv_path = os.path.join(vicon_dir, "leg_skeleton_data")
    csv_path = csv_path + ".csv"

    # load csv
    vicon_df = pd.read_csv(filepath_or_buffer=csv_path)
    if verbose:
        print('{} measurements detected.'.format(len(vicon_df)))
    stat_dict['vicon_len'] = len(vicon_df)

    return vicon_df, stat_dict

def detect_missing_vicon_intervals(vicon_df, stat_dict, labels_df, epsilon=1e-4, verbose=True):

    # convert timestamp to datetime, and extract numeric values from link columns
    vicon_df = convert_time_to_datetime(vicon_df)
    vicon_df, link_names, stat_dict = convert_vicon_data_to_numeric(data_df=vicon_df, stat_dict=stat_dict, verbose=verbose)
    vicon_df, link_names_separated = separate_link_columns(data_df=vicon_df, link_names=link_names, verbose=verbose)

    # mark lost vicon intervals
    vicon_df['lost_vicon'] = vicon_df[link_names_separated].apply(lambda x: 1 if abs(x).mean() < epsilon else 0, axis=1)
    vicon_df['label'] = labels_df['label']
    vicon_df['lost_vicon'] = vicon_df.apply(lambda x: 0 if x['lost_vicon'] and x['label'] > 0 else x['lost_vicon'], axis=1)

    # detect lost vicon intervals if bigger than 1.5 second by iterating over dataframe
    lost_vicon_intervals = []
    open_interval = False
    for i in range(1, len(vicon_df)):
        if i < int(len(vicon_df) * 0.75) and not open_interval and vicon_df.iloc[i]['lost_vicon'] == 1:
            open_interval = True
            lost_vicon_intervals.append([i])
        elif open_interval and vicon_df.iloc[i]['lost_vicon'] == 0 and (vicon_df.iloc[i]['datetime'] - vicon_df.iloc[lost_vicon_intervals[-1][0]]['datetime']).seconds >= 1.5:
            open_interval = False
            lost_vicon_intervals[-1].append(i)
        elif open_interval and vicon_df.iloc[i]['lost_vicon'] == 0 and (vicon_df.iloc[i]['datetime'] - vicon_df.iloc[lost_vicon_intervals[-1][0]]['datetime']).seconds < 1.5:
            # remove last object in intervals list
            open_interval = False
            lost_vicon_intervals.pop()

    if len(lost_vicon_intervals) > 0:
        total_seconds = 0
        for delta in [vicon_df.iloc[y]['datetime'] - vicon_df.iloc[x]['datetime'] for x,y in lost_vicon_intervals]:
            total_seconds += delta.total_seconds()
        stat_dict['lost_vicon_seconds'] = total_seconds
        stat_dict['lost_vicon_percentage'] = total_seconds / (vicon_df.iloc[-1]['datetime'] - vicon_df.iloc[0]['datetime']).total_seconds()
        stat_dict['lost_vicon_intervals'] = lost_vicon_intervals
        if verbose:
            print(f'{len(lost_vicon_intervals)} lost Vicon intervals detected.')
            print(f'Total lost time: {total_seconds} seconds.')
            print(f'Lost time percentage: {total_seconds / (vicon_df.iloc[-1]["datetime"] - vicon_df.iloc[0]["datetime"]).total_seconds()}')
    else:
        stat_dict['lost_vicon_seconds'] = 0
        stat_dict['lost_vicon_percentage'] = 0
        stat_dict['lost_vicon_intervals'] = []
        if verbose:
            print('No lost Vicon intervals detected.')
        
    # remove lost vicon intervals
    labels_df = labels_df[~vicon_df.index.isin([i for interval in lost_vicon_intervals for i in range(interval[0], interval[1])])]

    return labels_df, stat_dict 

# load labels csv as dataframe
def load_labels_data(data_dir, stat_dict, verbose=True):

    # create csv path
    csv_path = os.path.join(data_dir, "labels_data")
    csv_path = csv_path + ".csv"

    # load csv
    labels_df = pd.read_csv(filepath_or_buffer=csv_path)
    if verbose:
        print('{} labels detected.'.format(len(labels_df)))
    stat_dict['labels_len'] = len(labels_df)

    return labels_df, stat_dict

# load IMU data csv as dataframe
def load_imu_data(data_dir, stat_dict, verbose=True):

    # create csv path
    csv_path = os.path.join(data_dir, "imu_data")
    csv_path = csv_path + ".csv"

    # load csv
    imu_df = pd.read_csv(filepath_or_buffer=csv_path)
    if verbose:
        print('{} IMU measurements detected.'.format(len(imu_df)))
    stat_dict['imus_len'] = len(imu_df)

    # convert timestamp to datetime format
    imu_df['datetime'] = imu_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    imu_df['datetime'] = pd.to_datetime(imu_df['datetime'])

    return imu_df, stat_dict

# save processed dataframe in h5 format
def save_processed_h5(data_df, output_dir, data_dir, verbose=True):

    # create target file name as {category_name}_{enrollment_name}_{session_name}.h5
    subject_name = data_dir.split('/')[-3]
    session_name = data_dir.split('/')[-2]
    target_file_name = subject_name + '_' + session_name + '.h5'

    # save as h5
    data_df.to_hdf(os.path.join(output_dir, target_file_name), key='df', mode='w')
    if verbose:
        print('{} samples stored'.format(data_df.shape[0]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path to folder containing data files', default='/Data/imugr/datasets/bracelet/ID07/standing_gestures34free/', type=str)
    parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='/Data/imugr/datasets/bracelet/prepared_data', type=str)
    args = parser.parse_args()


    prepare_session_data(data_dir=args.data_dir, output_dir=args.output_dir)