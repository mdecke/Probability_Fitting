import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def get_state(x: np.array):
    theta = np.arctan2(x[1], x[0])
    theta_dot = x[2]
    return np.array([theta, theta_dot])

def augment_data(data:pd.DataFrame):
    data['prev_action'] = data['actions'].shift(1, fill_value=0)
    data['prev_state'] = data['states'].shift(1, fill_value=0)

    mask = data['episode'] != data['episode'].shift(1)
    data.loc[mask, 'prev_action'] = np.array(0)
    data.loc[mask, 'prev_state'] = data.loc[mask, 'prev_state'].apply(lambda _: [0, 0, 0])


    return data

def get_nparray(x):
    x_list = []
    for i in range(len(x)):
        x_list.append(x[i])
    x_np = np.array(x_list)
    return x_np


def make_input(obs:np.ndarray, prev_action:np.ndarray=None, prev_obs:np.ndarray=None):
        theta = np.arctan2(obs[0,1], obs[0,0])
        theta_dot = obs[0,2]
        new_state = np.array([theta, theta_dot]).reshape(1, -1)
        if prev_action == None and prev_obs == None:
            input_model = new_state
        elif prev_action.all() != None and prev_obs == None:
            input_model = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
        elif prev_action.all() != None and prev_obs.all() != None:
            theta_prev = np.arctan2(prev_obs[0,1], prev_obs[0,0])
            theta_dot_prev = prev_obs[0,2]
            new_prev_state = np.array([theta_prev, theta_dot_prev]).reshape(1, -1)
            intermed = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
            input_model = np.concatenate((intermed, new_prev_state), axis=1)
        else:
            raise ValueError("Invalid input type. Must be 'state' or 'state_action' or 'prev_state_action'")
        return input_model
   