#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:35:19 2023

@author: alutes
"""
### Load torch modules
import torch
from torch import nn

class DubNet(nn.Module):
    def __init__(self,
                 n_filters = 20,
                 n_hidden_units = 20,
                 output_size = 2
                 ):
        super(DubNet, self).__init__()
        self.n_filters = n_filters
        self.n_hidden_units = n_hidden_units
        self.output_size = output_size

        # Use padless convolutions to shrink 4X4 -> 1X1
        self.conv1 = nn.Conv2d(1, n_filters, 2,padding=0) 
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters, 2, padding=0) 
        self.conv3 = nn.Conv2d(self.n_filters, self.n_filters, 2, padding=0)

        # Use Max pooling on each layer as well to find interesting low level patterns
        ## Fill in later
        
        # Send it all to a linear layer
        self.fc1 = nn.Linear(self.n_filters, self.n_hidden_units)
        self.fc2 = nn.Linear(self.n_hidden_units, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output_transform = nn.Softmax(dim = 0)

    def forward(self, input_mat):
        
        # Conv -> 1X1
        x = self.relu(self.conv1(input_mat)) # No padding, 4X4 input -> 3X3
        x = self.relu(self.conv2(x)) # 3X3 -> 2X2
        x = self.relu(self.conv3(x)) # 2X2 -> 1X1

        # Max Pooloing to Linear
        x = x.view(-1, self.n_filters)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1)
        x = self.output_transform(x)
        return x
