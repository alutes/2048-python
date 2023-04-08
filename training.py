#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:24:35 2023

@author: alutes
"""


from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from modeling import DubNet

# Converts a 4X4 numpy matrix into a torch matrix used for modeling
def np_to_torch(game):
    mat = torch.from_numpy(np.flip(game,axis=0).copy()).reshape([1,1,4,4]).float()
    return mat

# Convert string storing game matrix
def string_to_mat(string):
    # Load as numpy matrix
    np_mat = np.matrix(string).reshape([4,4])
    return np_to_torch(np_mat)

# Convert a lookahead value to a tensor
def output_val_to_tensor(labels):
    return torch.from_numpy(np.array([labels, 1-labels]))

# Train a model for a set number of epochs
def train(model, train_loader, criterion, optimizer, num_epochs, shuffle=True, print_every=500):
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0

        if shuffle:
            random.shuffle(train_loader)

        for i, data in enumerate(train_loader):
            # Get the inputs and labels
            inputs, labels = data
            
            # Convert labels
            label_tensor = output_val_to_tensor(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs.reshape([1,2]), label_tensor.reshape([1,2]))
            loss.backward()
            
            if i % batch_size == 0:
                optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % print_every == (print_every-1):    # Print every N mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_every))
                epoch_losses.append((epoch, i, running_loss))
                running_loss = 0.0
    return epoch_losses

# Load Data
base_path = Path("/Users/alutes/Documents/Data/")
files = [f for f in base_path.glob('**/*.csv') if f.is_file()]
df = pd.concat([pd.read_csv(f) for f in files])
del df['Unnamed: 0']

# Load game matrices into tensors
df['mat'] = df.game.apply(string_to_mat)

# Transform quality values
vals = df['lookahead_value'].values
vals = 2*np.minimum(.5, np.maximum(vals,0.0))

# Data loader
loader = list(zip(
    df['mat'].values,
    vals
    ))

# Define a model, and training objects
model = DubNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
losses = train(
      model=model,
      train_loader=loader,
      criterion=criterion,
      optimizer=optimizer,
      num_epochs=30,
      print_every=5000
      )


# Analyze Losses
loss_df = pd.DataFrame(losses, columns=['epoch', 'iteration', 'loss'])
plt.plot(loss_df['epoch'] + (loss_df['iteration'] / 50000), loss_df['loss'])
