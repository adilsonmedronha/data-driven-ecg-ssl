
import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.fft as fft
import torch.nn.functional as F
from utils import utils, analysis

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.Series2Vec.soft_dtw_cuda import SoftDTW
from models.Series2Vec.fft_filter import filter_frequencies


class BaseTrainer(object):

    def __init__(self, model, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']
        self.problem = config['problem']

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)

class S2V_SS_Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(S2V_SS_Trainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.train_loader):
            X, _, IDs = batch

            Distance_out, Distance_out_f, rep_out, rep_out_f = self.model.Pretrain_forward(X.to(self.device))
            '''
            y = rep_out - rep_out.mean(dim=0)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_y))
            cov_y = (y.T @ y) / (len(rep_out) - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])
            '''

            mask = torch.tril(torch.ones_like(Distance_out), diagonal=-1).bool()
            Distance_out = torch.masked_select(Distance_out, mask)
            Distance_out = Distance_normalizer(Distance_out)
            Distance_out_f = torch.masked_select(Distance_out_f, mask)
            Distance_out_f = Distance_normalizer(Distance_out_f)
            Dtw_Distance = cuda_soft_DTW(self.sdtw, X, len(X))
            Dtw_Distance = Distance_normalizer(Dtw_Distance)
            X_f = filter_frequencies(X)
            Euclidean_Distance_f = Euclidean_Dis(X_f, len(X_f))
            Euclidean_Distance_f = Distance_normalizer(Euclidean_Distance_f)
            temporal_loss = F.smooth_l1_loss(Distance_out, Dtw_Distance)
            frequency_loss = F.smooth_l1_loss(Distance_out_f, Euclidean_Distance_f)

            total_loss = temporal_loss + frequency_loss
            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()
            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        return self.epoch_metrics