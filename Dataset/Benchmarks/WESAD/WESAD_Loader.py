import os
import logging
import requests
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch

logger = logging.getLogger(__name__)
# Main function for downloading and processing the WISDM Datasets
# Returns the train and test sets


def WESAD():

    current_path = os.getcwd()
    train_data_path = os.path.join(current_path, 'Datasets/WESAD/WESAD_train.pt')
    test_data_path = os.path.join(current_path, 'Datasets/WESAD/WESAD_test.pt')
    
    Data = {}
    Data_train = torch.load(train_data_path)
    Data_test = torch.load(test_data_path)

    Data['train_data'] = Data_train['samples'].numpy()
    Data['train_label'] = Data_train['labels'].numpy()

    Data['test_data'] = Data_test['samples'].numpy()
    Data['test_label'] = Data_test['labels'].numpy()


    logger.info("{} samples will be used for training".format(len(Data['train_label'])))
    logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    return Data


def load_activity(df):
    norm = True
    verbose = 1
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['label'])
    all_series = df.series.unique()
    train_series, test_series = train_test_split([x for x in range(len(all_series))], test_size=22, random_state=1)
    train_series = all_series[train_series]
    test_series = all_series[test_series]

    train_data = np.empty((0, 3))
    train_label = np.empty(0)
    print("[Data_Loader] Loading Train Data")
    for series in train_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        train_data = np.vstack((train_data, series_data))
        train_label = np.hstack((train_label, series_labels))

    test_data = np.empty((0, 3))
    test_label = np.empty(0)
    print("[Data_Loader] Loading Test Data")
    for series in test_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        test_data = np.vstack((test_data, series_data))
        test_label = np.hstack((test_label, series_labels))
    return train_data, train_label, test_data, test_label