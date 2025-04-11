import os
import numpy as np

import torch
from torch.utils.data import Dataset

from .augmentations import DataTransform



class Load_Dataset(Dataset):
    '''Custom class representing a augmented ``Dataset``.
    '''
    # Initialize your data, download, etc.
    def __init__(self, X_train, y_train, config,  training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if y_train is None:
            y_train = torch.zeros((X_train.shape[0],1))

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            print("Augmenting the dataset...")
            self.aug1, self.aug2 = DataTransform(self.x_data, config)
            print("")

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"), weights_only=True)
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"), weights_only=True)
    test_dataset = torch.load(os.path.join(data_path, "test.pt"), weights_only=True)

    train_dataset = Load_Dataset(train_dataset, configs['aug'], training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs['aug'], training_mode)
    test_dataset = Load_Dataset(test_dataset, configs['aug'], training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs['batch_size'],
                                               shuffle=True, drop_last=configs['drop_last'],
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs['batch_size'],
                                               shuffle=False, drop_last=configs['drop_last'],
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs['batch_size'],
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader