#!/usr/bin/env python3
import numpy as np
import os
import torch
import torch.autograd as grad
import torch.utils.data as Data


class PkBrain(torch.nn.Module):
    """ superclass for compatibility """
    def __init__(self, train_data, train_params):
        super().__init__()
        # data only gets passed to determine its shape
        self.nb_inchannels = train_data.shape[1]
        self.output_size = train_params.shape[-1]

class starter_twos(PkBrain):
    def __init__(self, train_data, yvals,):
        super().__init__(train_data, yvals,)
        ### Layers
        self.conv1 = torch.nn.Conv1d(self.nb_inchannels, 8, kernel_size=(21,), stride=1, padding=0)
        self.mp4 = torch.nn.MaxPool1d(4, padding = 2)
        self.conv2 = torch.nn.Conv1d(8, 8, kernel_size=(23,), stride=1, padding=2)
        self.mp2 = torch.nn.MaxPool1d(4, padding=2)
        self.fla = torch.nn.Flatten()
        self.ufl = torch.nn.Unflatten(1, torch.Size([1, 2752]))
        self.l1 = torch.nn.Linear(2752, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.relu = torch.nn.ReLU()
        self.lfinal = torch.nn.Linear(128, self.output_size)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.mp4(x)
        x = self.conv2(x)
        x = self.mp4(x)
        x = self.mp2(x)
        x = self.fla(x)
        x = self.ufl(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.lfinal(x)
        return x
