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
import torch.nn.functional as F
import numpy as np
import random

from modeling import DubNet

from matplotlib import pyplot as plt

class SoftCrossEntropyLoss(nn.Module):
   def __init__(self, weights = torch.tensor(1.0)):
      super(SoftCrossEntropyLoss, self).__init__()
      self.weights = weights

   def forward(self, y_hat, y):
      p = F.log_softmax(y_hat, 1)
      w_labels = self.weights*y
      loss = -(w_labels*p).sum() / (w_labels).sum()
      return loss

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
    return torch.from_numpy(np.array([labels, 1-labels])).reshape(1,2)


def get_accuracy(outputs, batch_labels, hinge=.5):
    out = outputs[:,0].detach().numpy() 
    lab = batch_labels[:,0].detach().numpy()
    accuracy = ((out < hinge) == (lab < hinge)).mean()
    return accuracy
    

def plot_accuracy_soft(out, lab):
    # Scatterplot    
    plt.scatter(out, lab, alpha = .1)

def plot_accuracy(out, lab):
    # hists    
    bins = np.arange(0,1,.01)
    plt.hist(out[lab == 0.0], density = True, histtype = 'step', cumulative = True, bins=bins)
    plt.hist(out[lab == 1.0], density = True, histtype = 'step', cumulative = True, bins=bins)

def print_progress(epoch, loss, i, batch_input, batch_labels, outputs, accuracy, show_plots=False):
    # Transform results
    out = outputs[:,0].detach().numpy() 
    lab = batch_labels[:,0].detach().numpy()
    print('[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, loss, accuracy))
    
    if show_plots:
        # Plot Example Matrix
        index = np.random.randint(0, len(batch_input) - 1)
        example_input = batch_input[index]
        example_output = outputs[index]
        example_label= batch_labels[index] 
        
        plt.subplot(1, 2, 1)
        plot_matrix(example_input, example_label, example_output)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plot_accuracy(out, lab)
        plt.show()
    
def plot_matrix(example_input, example_label, example_output):
    game = example_input.numpy().reshape(4,4)
    plt.imshow(np.log1p(game))
    for i in range(4):
        for j in range(4):
            text = plt.text(j, i, int(game[i, j]), ha="center", va="center", color="w")
    
    output_print = round(example_output[0].item(), 2)
    label_print = round(example_label[0].item(), 2)
    plt.title(f"""label: {label_print}
              prediction: {output_print}""")


# Train a model for a set number of epochs
def train(model, train_loader, criterion, optimizer, num_epochs, shuffle=True, batch_size=500):
    loss_list = []
    for epoch in range(num_epochs):

        if shuffle:
            random.shuffle(train_loader)

        input_list = []
        label_list = []
        for i, data in enumerate(train_loader):
            # Get the inputs and labels
            inputs, labels = data
            
            input_list.append(inputs)
            label_list.append(output_val_to_tensor(labels))
            
            if i % batch_size == (batch_size - 1) or i == (len(train_loader) - 1):
                current_batch_size = len(input_list)
                batch_input = torch.cat(input_list)
                batch_labels = torch.cat(label_list)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                outputs = model(batch_input)
                loss = criterion(outputs, batch_labels)
                
                # Backprop
                loss.backward()
                optimizer.step()
                
                # Get Accuracy
                accuracy = get_accuracy(outputs, batch_labels, hinge=.5)
                
                # Losses
                loss_list.append({
                    'epoch' : epoch, 
                    'i' : i, 
                    'loss' : loss.item(),
                    'accuracy' : accuracy
                    })
                
                # Print Progress  
                if np.random.rand()<.01:
                    print_progress(epoch, loss.item(), i, batch_input, batch_labels, outputs, accuracy, show_plots=True)

                # reset the batch
                input_list = []
                label_list = []
    return loss_list


# Load Data
base_path = Path("/Users/alutes/Documents/Data/")
files = [f for f in base_path.glob('**/*.csv') if f.is_file()]
df = pd.concat([pd.read_csv(f) for f in files])

# Load game matrices into tensors
df['mat'] = df.game.apply(string_to_mat)

# Transform quality values
episode_max = df.groupby('episode')['move'].max()
df['episode_max'] = df['episode'].map(episode_max)
df['moves_remaining'] = df['episode_max'] - df['move']
df['target'] = df['moves_remaining'] < 100

vals = df['lookahead_value'].values
vals = 2*np.minimum(.5, np.maximum(vals,0.0))
df['val'] = vals

# Class imbalance ratio
relative_weight = 1 / np.mean(df.target)
relative_weight = 1 / 10.0

# Data loader
train_loader = list(zip(
    df['mat'].values,
    df['target'].values * 1.0
    ))

# Define a model, and training objects
model = DubNet(
    n_filters = [20, 20],
    n_hidden_units = 100
    )
criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, relative_weight]))
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
loss_list = train(
      model=model,
      train_loader=train_loader,
      criterion=criterion,
      optimizer=optimizer,
      num_epochs=600,
      batch_size=800
      )


# Analyze Losses
loss_df = pd.DataFrame(loss_list)
plt.plot(loss_df['epoch'] + (loss_df['i'] / loss_df['i'].max()), loss_df['loss'])
plt.plot(loss_df['epoch'] + (loss_df['i'] / loss_df['i'].max()), loss_df['accuracy'])
