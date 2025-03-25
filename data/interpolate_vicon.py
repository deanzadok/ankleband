import os
import sys
sys.path.append('.')
import time
import glob
import argparse
import datetime
import pandas as pd
import numpy as np

def interpolate_vicon_data(data_dir, method='linear', verbose=True):

    # load vicon csv as dataframe
    stat_dict = {}
    data_df, stat_dict = load_vicon_csv(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

    # convert timestamp to datetime format
    data_df = convert_time_to_datetime(data_df)

    # convert to numeric values
    data_df, link_names, stat_dict = convert_vicon_data_to_numeric(data_df=data_df, stat_dict=stat_dict, verbose=verbose)

    # for each link, separate xyz into 3 separated columns
    data_df, link_names_separated = separate_link_columns(data_df, link_names, verbose=verbose)

    # put NaN in cells with detached markers (locations with [0,0,0])
    data_df = replace_detached_markers_with_nan(data_df, link_names_separated, stat_dict=stat_dict, verbose=verbose)

    # fill values of missing markers
    data_df = interpolate_missing_markers(data_df=data_df, link_names_separated=link_names_separated, method=method, verbose=verbose)

    # merge separated link names back to 3-d positions
    data_df = merge_separated_links(data_df, link_names, link_names_separated, verbose=verbose)

    # converting back to "(x; y; z)"
    data_df = convert_vicon_data_to_string(data_df, link_names, verbose=verbose)

    # saving the modified version
    save_vicon_data(data_df, data_dir, verbose=verbose)
    
    return stat_dict

# load vicon csv as dataframe
def load_vicon_csv(data_dir, stat_dict, verbose=True):

    # create csv path
    vicon_dir = os.path.join(data_dir, "vicon")
    csv_path = os.path.join(vicon_dir, "leg_skeleton_data")
    csv_path = csv_path + ".csv"

    # load csv
    data_df = pd.read_csv(filepath_or_buffer=csv_path)
    if verbose:
        print('{} measurements detected.'.format(len(data_df)))
    stat_dict['vicon_len'] = len(data_df)

    return data_df, stat_dict

# convert timestamp to datetime format
def convert_time_to_datetime(data_df):

    data_df['datetime'] = data_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data_df['datetime'] = pd.to_datetime(data_df['datetime'])
    return data_df

# convert measurements to numeric values and remove nans
def convert_vicon_data_to_numeric(data_df, stat_dict, verbose=True):

    # get link names and other columns
    link_names = get_link_names(data_df=data_df)

    # convert cells of vicon dataframe to numeric
    data_df, stat_dict = convert_vicon_cells_to_numeric(data_df=data_df, column_names=link_names, stat_dict=stat_dict, verbose=verbose)

    return data_df.dropna(), link_names, stat_dict

# differentiate between link names and other columns and return the two lists
def get_link_names(data_df):

    # get link names
    link_names = []
    for column_name in data_df.columns:
        if 'leg_skeleton' in column_name:
            link_names.append(column_name)
    link_names = data_df[link_names].columns.values

    return link_names

# convert cells of vicon dataframe to numeric
def convert_vicon_cells_to_numeric(data_df, column_names, stat_dict, verbose=True, stat_label='vicon_nan_count'):

    # filter format "(x; y; z)" to [x, y, z]
    data_df[column_names] = data_df[column_names].applymap(lambda x: np.array(x.split('(')[-1].split(')')[0].replace(' ','').split(';'), dtype=np.float64))
    nan_values_count = data_df.isna().sum(axis=1).sum()
    if verbose:
        print('{} NaN values detected in Vicon dataframe.'.format(nan_values_count))
    stat_dict[stat_label] = nan_values_count

    return data_df.dropna(), stat_dict

# for each link, separate xyz into 3 separated columns
def separate_link_columns(data_df, link_names, verbose=True):

    if verbose:
        print('Separating links columns...')

    for link_name in link_names:
        data_df[link_name + '_x'] = data_df[link_name].apply(lambda x: x[0])
        data_df[link_name + '_y'] = data_df[link_name].apply(lambda x: x[1])
        data_df[link_name + '_z'] = data_df[link_name].apply(lambda x: x[2])
    data_df = data_df.drop(columns=link_names)
    # set separated joints list
    link_names_separated_lst = [[x+'_x',x+'_y',x+'_z'] for x in link_names.tolist()]
    link_names_separated = np.array(link_names_separated_lst).flatten()

    return data_df, link_names_separated

# put NaN in cells with detached markers (locations with [0,0,0])
def replace_detached_markers_with_nan(data_df, link_names_separated, stat_dict, verbose=True):

    data_df[link_names_separated] = data_df[link_names_separated].applymap(lambda x: x if abs(x) > 1 else np.nan)
    detached_markers_num = int(data_df.isnull().values.sum()/3)
    if verbose:
        print('Found approximately {} detached markers.'.format(detached_markers_num))
    stat_dict['detached_markers'] = detached_markers_num

    return data_df

# fill values of missing markers
def interpolate_missing_markers(data_df, link_names_separated, method, verbose=True):

    if verbose:
        print('Interpolating missing markers recordings...')
    
    if method == 'time':
        df_interpolated = data_df.set_index('datetime').interpolate(method=method)
        return df_interpolated.reset_index()
    else:
        data_df[link_names_separated] = data_df[link_names_separated].interpolate(method=method)
        return data_df

# merge separated link names back to 3-d positions
def merge_separated_links(data_df, link_names, link_names_separated, verbose=True):

    if verbose:
        print("Merging separated links back...")
    
    for link_name in link_names:
        data_df[link_name] = data_df.apply(lambda x: np.array([x[link_name+'_x'],x[link_name+'_y'],x[link_name+'_z']]) , axis=1)
    return data_df.drop(columns=link_names_separated)

# convert measurements back to string values as "(x; y; z)"
def convert_vicon_data_to_string(data_df, link_names, verbose=True):

    if verbose:
        print('Converting vicon data to string format...')

    data_df[link_names] = data_df[link_names].applymap(lambda x: np.array2string(x, separator="; ").replace("[","(").replace("]",")"))
    return data_df

# save modified version of a vicon csv
def save_vicon_data(data_df, data_dir, verbose=True):

    if verbose:
        print("Saving modified dataframe...")
    
    vicon_dir = os.path.join(data_dir, "vicon")
    csv_path = os.path.join(vicon_dir, "leg_skeleton_data")
    csv_path = csv_path + "_modified.csv"

    data_df.to_csv(path_or_buf=csv_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path to folder containing data files', default='/Data/imugr/datasets/bracelet/ID07/standing_gestures12', type=str)
    args = parser.parse_args()


    interpolate_vicon_data(data_dir=args.data_dir)