#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:19:53 2023

@author: alutes
"""
import numpy as np
import tqdm
import pandas as pd

from value_functions import value_medium, value_fast, value_slow
from logic import new_game, game_over, all_moves_dict, add_block, reverse
from search_tree import choose_max, expand_and_prune, score_move_space

import constants as c
from scipy.stats import poisson as ps

# Sample over random boards that look like we might get a few thousand moves in
def sample_board(base_rate = 11, value_length = c.nrow * c.ncol):
    
    # Get a steadily decreasing 1 -> 0 set of values
    cum_quantiles = np.cumprod(np.random.rand(value_length))    
    
    # Modulate each quantile by some 0-1 %
    quantiles = np.random.rand(value_length)    
    
    # Values are based on quantiles from a poisson distribution
    values = ps.ppf(quantiles * cum_quantiles, mu = base_rate)
    
    # Transform into a matrix
    mat = np.reshape(values, [c.nrow, c.ncol])
    for row_index in range(1, c.nrow, 2):
        mat[row_index,:] = reverse(mat[row_index,:])
    
    mat = ((2**mat) * (mat > 0)).astype(int)
    
    # Randomize orientation over 8 possible flips and rotations
    if np.random.rand() > .5:
        mat = mat.T
    if np.random.rand() > .5:
        mat = np.flip(mat, axis = 0)
    if np.random.rand() > .5:
        mat = np.flip(mat, axis = 1)

    return mat


# Configurations
config_choices = [    
    (3, 2, value_medium),
    (3, 2, value_slow)
]



# Static Parameters
max_initial_states = 16
max_retained_actions = 2
max_moves_per_episode = 10000

# Logging
state_data = []
log_every = 10

# Play N games
num_epsiodes = 1000
for episode_index in range(0, num_epsiodes):
    print(f"#####\n Episode: {episode_index} \n#####")

    # Choose a reasonable depth and breadth
    depth, max_retained_states, value_fn = config_choices[np.random.randint(len(config_choices))]
    print(depth, max_retained_states, value_fn.__qualname__)

    # Instantiate game using our board sampler
    game = sample_board()

    for move_index in tqdm.tqdm(range(max_moves_per_episode)):

        # Loop through each action
        move_space = expand_and_prune(
            game,
            depth = depth,
            value_fn = value_medium,
            max_initial_states = max_initial_states,
            max_retained_actions = max_retained_actions,
            max_retained_states = max_retained_states
        )

        # Evaluate Actions
        action_values = {}
        for i, action_name in enumerate(move_space):        

            # Calculate Expected Board Value
            action_values[action_name] = score_move_space(move_space[action_name], value_fn, discount = 1)

        # Find the Nth highest action value
        actions_sorted_by_value = sorted(
            list(action_values.keys()), 
            key = action_values.get, 
            reverse = True
            )

        # unpack action
        best_action_name = actions_sorted_by_value[0]
        move = all_moves_dict[best_action_name]
        action_game, done = move(game)
        lookahead_value = action_values[best_action_name]

        # Choose Action
        move = choose_max(action_values)
        game, done = all_moves_dict[move](game)
        if not done:
            raise ValueError("Illegal move chosen")
        game = add_block(game)
        
        # Log Data
        is_game_over = game_over(game)
        if move_index%log_every == 0 or is_game_over:
            state_data.append({
                'episode' : episode_index,
                'move' : move_index,
                'depth' : depth,
                'max_retained_states' : max_retained_states,
                'max_retained_actions' : max_retained_actions,
                'lookahead_value' : lookahead_value,
                'value_fn' : value_fn.__qualname__,
                'total' : game.sum(),
                'game' : game
            })
            
        # End the game if it's over
        if is_game_over:
            df = pd.DataFrame(state_data)
            df.to_csv(f"/Users/alutes/Documents/Data/state_data_{episode_index}.csv", index = False)
            print(game.sum())
            state_data = []
            break