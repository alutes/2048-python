#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:35:19 2023

@author: alutes
"""
### Load torch modules
import torch
from torch import nn
import numpy as np

class DubNet(nn.Module):
    def __init__(self,
                 n_filters = [20, 20],
                 n_hidden_units = 100,
                 output_size = 2,
                 input_size = [4,4]
                 ):
        super(DubNet, self).__init__()
        self.n_filters = n_filters # lists the number of filters for each layer
        self.n_hidden_units = n_hidden_units
        self.output_size = output_size

        # Use  convolutions to shrink 4X4 -> 2X2
        self.conv1 = nn.Conv2d(1, self.n_filters[0], 2, padding=0) 
        self.conv2 = nn.Conv2d(self.n_filters[0], self.n_filters[1], 2, padding=0) 

        # The size of the output should be 2X2 times the number of filters applied
        self.final_input_size_flattened = np.prod([a - 2 for a in input_size]) * self.n_filters[1]
        
        # Flatten and send it all to a linear layer
        self.fc1 = nn.Linear(self.final_input_size_flattened, self.n_hidden_units)
        
        # Final Linear Layer to outputs
        self.fc2 = nn.Linear(self.n_hidden_units, self.output_size)
        
        # Standard functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output_transform = nn.Softmax(dim = 0)

    def pre_process(self, input_mat):
        input_transformed = torch.clamp(input_mat, min = 1.0)
        input_transformed = torch.log2(input_transformed)
        return input_transformed
        
    def forward(self, input_mat):
        
        # Transform Matrix
        x = self.pre_process(input_mat)
        
        # Conv -> 1X1
        x = self.relu(self.conv1(x)) # No padding, 4X4 input -> 3X3
        x = self.relu(self.conv2(x)) # 3X3 -> 2X2

        # Max Pooloing to Linear
        x = x.view(-1, self.final_input_size_flattened)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.output_transform(x)
        return x