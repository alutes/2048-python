#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:24:09 2023

@author: alutes
"""
from matplotlib import pyplot as plt
import tqdm
import random
import time
import numpy as np
import copy

import constants as c
from logic import all_moves, add_block, new_game, game_over
from value_functions import (smoothness, monotonicity_cost, corner_value, free_spaces, 
                            VALUE_MODEL, value_manual_score, monotonicity, ai, choose_one_random)
from search_tree import random_move, build_move_states

####################
# Time Functions so we can optimize efficiency where needed
####################
n = 1000
total_time = 0
game = new_game()
for i in range(n):
    print(i)
    done = False
    while not done:
        move =  random_move()
        start = time.time()
        game, done = move(game)
        end = time.time()
    game = add_block(game)
    if game_over(game):
        break
    total_time += (end - start)
print(i)
print(i / total_time)


####################
# Print Out Cost Functions and boards
####################
corner_weight = VALUE_MODEL['corner_weight']
monotonicity_weight = VALUE_MODEL['monotonicity_weight']
smooth_weight = VALUE_MODEL['smooth_weight']
free_weight = VALUE_MODEL['free_weight']
plot = True

import tqdm

game = new_game()
for move_index in tqdm.tqdm(range(1000)):
    plot = move_index%50==0

    if plot:
        fig, axs = plt.subplots(2, 2)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=1.2)

    # Loop through each action
    move_space = expand_states(game, depth = 2, max_states_retained = 6)

    action_values = {}
    for i, action_name in enumerate(move_space):

        # unpack action
        move = all_moves_dict[action_name]
        action_game, done = move(game)
        lookahead_value = score_move_space(move_space[action_name])
        action_values[action_name] = lookahead_value

        # Value action
        score = value_manual_score(action_game)

        # Scores
        mat = action_game # try out different functions of the matrix as input
        scores = [
            (monotonicity(mat), 0.0),
            (corner_value(mat), corner_weight),
            (smoothness(mat), smooth_weight),
            (free_spaces(mat), free_weight)
            ]

        if plot:
            # Title
            title = f"{action_name}: {np.round(lookahead_value,2)}, {np.round(score,2)} \n ("
            for s,_ in scores:
               title += str(np.round(s, 2)) + ",  "
            title += ")"

            # plot area
            ax = axs[i%2, int(i / 2)]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(action_game)
            for i in range(4):
                for j in range(4):
                    text = ax.text(j, i, int(action_game[i, j]), ha="center", va="center", color="w")
            ax.set_title(title)

    # Choose Action
    move = choose_max(action_values)
    game, done = all_moves_dict[move](game)
    if not done:
        raise ValueError("Illegal move chosen")
    game = add_block(game)
    if game_over(game):
        raise ValueError("Game Over")

    if plot:
        plt.suptitle(move, fontsize = 20)
        plt.show()



####################
# Time different search depths
####################


import numpy as np
import tqdm
import pandas as pd

from value_functions import value_medium, value_fast, value_slow
from logic import new_game, game_over, all_moves_dict, add_block
from search_tree import choose_max, expand_states, score_move_space

import time
# Configurations
config_choices = [
    (0, 28, value_fast),
    (0, 28, value_medium),
    (0, 28, value_slow),

    (1, 15, value_fast),
    (1, 15, value_medium),
    (1, 10, value_slow),

    (2, 6, value_fast),
    (2, 3, value_medium),

    (3, 2, value_fast)
]


for config in config_choices:

    # Choose a reasonable depth and breadth
    depth, breadth, value_fn = config
    print(depth, breadth, value_fn.__qualname__)
    start = time.time()

    # Instantiate game
    game = new_game()

    for move_index in range(2):

        # Loop through each action
        move_space = expand_states(
            game,
            depth = depth,
            max_states_retained = breadth
            )

        # Evaluate Actions
        action_values = {}
        for i, action_name in enumerate(move_space):

            # unpack action
            move = all_moves_dict[action_name]
            action_game, done = move(game)
            lookahead_value = score_move_space(move_space[action_name], value_fn = value_fn)
            action_values[action_name] = lookahead_value
    end = time.time()
    print(end - start)
    



    
    
    


