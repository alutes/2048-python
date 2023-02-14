#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 20:01:07 2023

@author: alutes
"""
import numpy as np
import copy
from logic import all_moves, num_free_spaces, reverse, transpose
import constants as c

########################################
## Manual Value Functions
## Custom Formulae for fitness function which incorporates 4
##  qualities of a board matrix f
##
##      1. Number of free spaces sum_ij(f_ij = 0)
##
##      2. Smoothness: the norm of some scaling function of the gradient
##          |∇f| = |g([∂f/∂x, ∂f/∂y])| 
##
##      3. Monotonicity: Look for strictly increasing values 
##          in x and y directions
##
##      4. Keeping values in the corner. Imposes a cost on moving 
##          values to the middle where there is less flexibility
##
########################################

VALUE_MODEL = {
   'smooth_weight' : 1.0,           # relative weight of smoothness
   'monotonicity_weight' : 0.0,     # relative weight of monotonicity
   'free_weight' : 1.0,             # relative weight of free spaces
   'corner_weight' : 1.0,           # relative weight of keeping high values to corners
   'board_state_weight' : 1.0,      # relative value of the board state compared to not losing
   'not_loss_weight' : 1.0,         # relative weight of not losing after max play depth
   'depth_achived_weight' : 1.0     # relative weight of achieving % of max play depth before losing (must be less than not_loss_weight)
    }

cost_kernel = np.array([
    [0, 1, 1, 0],
    [1, 2, 2, 1],
    [1, 2, 2, 1],
    [0, 1, 1, 0]
])

# 0-1 score where 
#   1 = all value is in the corners
#   0 = all value is in the middle
def corner_value(game):
    return 1.0 - np.sum(game * cost_kernel) / np.sum(game * 2)

# % of spaces which are free
#  mean_ij(f_ij = 0)
def free_spaces(game):
    return num_free_spaces(game) / game.size

# 0-1 measure of smoothness
# 1 - |∇f| / |f|
# |∇f| = |g([∂f/∂x, ∂f/∂y])| 
def smoothness(game):
    return 1 - np.sum(np.gradient(game)) / (2 * np.sum(game))


# Gives cost to deviation from monoticity
def monotonicity_vec(vector):
    diff_vec = -np.diff(vector) # penalize any decreases
    return np.sum(
                    diff_vec * 
                    (diff_vec > 0) *  # sum of decreases only
                    (vector[:-1] > 0) # not including blank spaces
                ) 

# presents a single vector which zig zags through a vector in the following order
    # [16,      15,     14,     13],
    # [9,       10,     11,     12],
    # [8,       7,      6,      5],
    # [1,       2,      3,      4]
def zig_zag_mat(game):
    mat = copy.deepcopy(game)
    for i in range(1, mat.shape[0], 2):
        mat[i] = reverse(mat[i])
    return mat.flatten()

# Scores forwards and backwards monotonicity cost for a single direction of the matrix
def monotonicity_dir(game):
    vector = zig_zag_mat(game)
    return np.min([monotonicity_vec(vector), monotonicity_vec(reverse(vector))])

# cost for deviations from monotonicity
def monotonicity_cost(game):
    return np.min([
            monotonicity_dir(game),
            monotonicity_dir(transpose(game)),
            monotonicity_dir(reverse(game)),
            monotonicity_dir(reverse(transpose(game)))
        ])

# 0-1 measure of monotonicity
def monotonicity(game):
    return 1.0 - monotonicity_cost(game) / np.sum(game)

def weighted_average(value_weight_pairs):
    total = 0.0
    total_weight = 0.0
    for value, weight in value_weight_pairs:
        total += value * weight
        total_weight += weight
    return total / total_weight

def value_manual_score(
        game, 
        corner_weight = VALUE_MODEL['corner_weight'], 
        monotonicity_weight = VALUE_MODEL['monotonicity_weight'], 
        smooth_weight = VALUE_MODEL['smooth_weight'], 
        free_weight = VALUE_MODEL['free_weight']
        ):
    mat = game
    return weighted_average([
        #(monotonicity(mat), monotonicity_weight), # too slow atm, exclude
        (corner_value(mat), corner_weight), 
        (smoothness(mat), smooth_weight), 
        (free_spaces(mat), free_weight)
        ])


###
# Final Value Function
###

# A fitness function for any particular game state
# @TODO: Make this informed by the different types of value functions below
def value_function(game):
    return value_manual_score(game)

