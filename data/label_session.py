import os
import sys
sys.path.append('.')
import time
import glob
import argparse
import datetime
import pandas as pd
import numpy as np
import open3d
import tty
import termios
from data.interpolate_vicon import convert_vicon_data_to_numeric


def label_session_data(data_dir, start_idx=0, speed=1):

    # load vicon csv as dataframe
    stat_dict = {}
    data_df, stat_dict = load_vicon_csv(data_dir=data_dir, stat_dict=stat_dict)

    # locate labels dataframe or create one
    labels_df = load_labels_csv(data_dir=data_dir, data_df=data_df)
    writing_state = 0

    # exit to save zero labels for free sessions
    if '_free' in data_dir:
        save_labels_data(data_dir=data_dir, labels_df=labels_df)
        return
    
    # convert to numeric values
    data_df, link_names, stat_dict = convert_vicon_data_to_numeric(data_df=data_df, stat_dict=stat_dict)

    # get all possible lines and colors for visualization
    pc_colors_dict = get_links_colors()
    pc_colors = links_colors_to_list(pc_colors_dict=pc_colors_dict, link_names=link_names.tolist())
    lines_dict = get_pc_lines()
    lines_idxs, lines_colors = get_lines_and_colors_for_links(lines_dict=lines_dict, link_names=link_names.tolist())

    # visualize the vicon stream by iterating over rows
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='Vicon', width=1200, height=1200, left=1720, top=10)
    point_cloud = open3d.geometry.PointCloud()
    lines_set = open3d.geometry.LineSet()
    lines_set.lines = open3d.utility.Vector2iVector(lines_idxs)
    point_cloud.colors = open3d.utility.Vector3dVector(pc_colors)
    lines_set.colors = open3d.utility.Vector3dVector(lines_colors)

    # configure key input
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    x = 0
    i = start_idx
    rot_state = np.array([40.0, -310.0])
    #rot_state = np.array([-250.0, 100.0])
    # rot_state = np.zeros(2)

    # iterate over dataframe actively until ESC pressed
    #for index, row in data_df.iterrows():
    while x != chr(27): # ESC

        if i >= len(data_df):
            break

        sample_np = np.vstack(data_df.iloc[i][link_names].values)
        point_cloud.points = open3d.utility.Vector3dVector(sample_np)
        lines_set.points = point_cloud.points
        vis.add_geometry(point_cloud)
        vis.add_geometry(lines_set)

        # visulization rotation
        visc = vis.get_view_control()
        rot_state = get_instruction_rotate(inp_key=x, rot_state=rot_state)
        if rot_state[0] != 0 or rot_state[1] != 0:
            visc.rotate(rot_state[0], rot_state[1])
        visc = vis.get_view_control()
        visc.rotate(rot_state[0], rot_state[1])

        vis.update_geometry(point_cloud)
        # vis.update_geometry()
        vis.update_renderer()
        vis.poll_events()

        # get instruction from user
        # get writing request
        writing_request = get_writing_request(inp_key=x)
        if writing_request >= 0:
            writing_state = writing_request

        # get step
        x=sys.stdin.read(1)[0]
        req_step = get_instruction_step(inp_key=x)

        # write label
        if req_step > 0:
            labels_df.loc[i:i+req_step, 'label'] = writing_state

        i+=req_step

        #vis.clear_geometries()
        vis.remove_geometry(point_cloud)

        if i < len(data_df):
            print('{}/{} samples plotted. time: {}. Writing: {}'.format(i,len(data_df), data_df['time'].iloc[i], writing_state))
        #time.sleep(0.05)

    
    # save labels data
    save_labels_data(data_dir=data_dir, labels_df=labels_df)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)


# load vicon csv as dataframe
def load_vicon_csv(data_dir, stat_dict, verbose=True):

    # create csv path
    vicon_dir = os.path.join(data_dir, "vicon")
    csv_path = os.path.join(vicon_dir, "leg_skeleton_data")
    csv_path = csv_path + "_modified.csv"

    # load csv
    data_df = pd.read_csv(filepath_or_buffer=csv_path)
    if verbose:
        print('{} measurements detected.'.format(len(data_df)))
    stat_dict['vicon_len'] = len(data_df)

    return data_df, stat_dict

# load current labels dataframe if found
def load_labels_csv(data_dir, data_df, verbose=True):

    # create a new labels dataframe if not found
    labels_path = os.path.join(data_dir, "labels_data.csv")
    if not os.path.isfile(labels_path):
        labels_df = pd.DataFrame()
        labels_df[['time', 'datetime']] = data_df[['time', 'datetime']]
        labels_df['label'] = 0
        if verbose:
            print('Labels dataframe missing.')
        return labels_df
    else:
        # load csv
        labels_df = pd.read_csv(filepath_or_buffer=labels_path)
        if verbose:
            print('Labels dataframe loaded.')
        return labels_df

def get_links_colors():

    # set color per link
    pc_colors_dict = {'leg_skeleton1':[150,0,250],
                      'leg_skeleton2':[0,250,250],
                      'leg_skeleton3':[0,250,0],
                      'leg_skeleton4':[250,180,0],
                      'leg_skeleton5':[30,0,30],
                      'leg_skeleton6':[250,250,0]}
        
    return pc_colors_dict

# sort according to link names order
def links_colors_to_list(pc_colors_dict, link_names):

    return np.array([pc_colors_dict[key] for key in link_names]) / 255.0


# get list of lines. Format: [[src_vertex, dest_vertex], ...]
def get_pc_lines():

    # set edges (lines) for known nodes (vertices)
    lines_dict = {'leg_skeleton1-leg_skeleton2':[['leg_skeleton1','leg_skeleton2'],[200,200,200]],
                  'leg_skeleton2-leg_skeleton3':[['leg_skeleton2','leg_skeleton3'],[200,200,200]],
                  'leg_skeleton3-leg_skeleton4':[['leg_skeleton3','leg_skeleton4'],[200,200,200]],
                  'leg_skeleton4-leg_skeleton1':[['leg_skeleton4','leg_skeleton1'],[200,200,200]],
                  'leg_skeleton1-leg_skeleton5':[['leg_skeleton1','leg_skeleton5'],[200,200,200]],
                  'leg_skeleton2-leg_skeleton5':[['leg_skeleton2','leg_skeleton5'],[200,200,200]],
                  'leg_skeleton3-leg_skeleton5':[['leg_skeleton3','leg_skeleton5'],[200,200,200]],
                  'leg_skeleton4-leg_skeleton5':[['leg_skeleton4','leg_skeleton5'],[200,200,200]],
                  'leg_skeleton5-leg_skeleton6':[['leg_skeleton5','leg_skeleton6'],[200,200,200]]}

    return lines_dict

# return lines indices and colors for a given list of link_names. output will be sorted according to the given input
def get_lines_and_colors_for_links(lines_dict, link_names):

    # get available lines based on link names
    available_lines_dict = {key:val for key,val in lines_dict.items() if (val[0][0] in link_names and val[0][1] in link_names)}

    # get lines indices for all available lines
    lines_idxs = [[link_names.index(val[0][0]),link_names.index(val[0][1])] for key,val in available_lines_dict.items()]

    # get colors per available lines
    lines_colors = [val[1] for key,val in available_lines_dict.items()]

    return lines_idxs, np.array(lines_colors) / 255.0

# return rotation based on pressed key
def get_instruction_rotate(inp_key, rot_state):

    if inp_key == 'q':
        rot_state[0] += 10.0
    if inp_key == 'w':
        rot_state[0] -= 10.0
    if inp_key == 'r':
        rot_state[1] += 10.0
    if inp_key == 't':
        rot_state[1] -= 10.0
    if inp_key == 'e':
        rot_state = np.zeros(2)
    
    return rot_state

# return step based on pressed key
def get_instruction_step(inp_key):

    step1_size = 1
    step2_size = 4
    step3_size = 9
    step4_size = 31
    step5_size = 55

    if inp_key == 'a':
        return step1_size
    elif inp_key == 's':
        return step2_size
    elif inp_key == 'd':
        return step3_size
    elif inp_key == 'f':
        return step4_size
    elif inp_key == 'g':
        return step5_size
    elif inp_key == 'z':
        return -step1_size
    elif inp_key == 'x':
        return -step2_size
    elif inp_key == 'c':
        return -step3_size
    elif inp_key == 'v':
        return -step4_size
    elif inp_key == 'b':
        return -step5_size
    else:
        return 0

# return step based on pressed key
def get_writing_request(inp_key):

    if inp_key == '`':
        return 0
    elif inp_key == '1' or inp_key == '2' or inp_key == '3' or inp_key == '4':
        return int(inp_key)
    else:
        return -1

# save labels dataframe as csv
def save_labels_data(data_dir, labels_df):

    csv_path = os.path.join(data_dir, "labels_data.csv")
    labels_df.to_csv(path_or_buf=csv_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path to folder containing data files', default='/Data/imugr/datasets/bracelet/ID07/standing_gestures34free', type=str)
    parser.add_argument('--start_idx', '-start_idx', help='starting index for the labelling session', default=0, type=int)
    args = parser.parse_args()


    label_session_data(data_dir=args.data_dir, start_idx=args.start_idx)


