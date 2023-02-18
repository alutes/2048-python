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
from logic import new_game, game_over, all_moves_dict, add_block
from search_tree import choose_max, expand_states, score_move_space

# Configurations
config_choices = [
    (0, 28, value_fast),
    (0, 28, value_medium),
    (0, 28, value_slow),

    (1, 15, value_fast),
    (1, 15, value_medium),
    (1, 10, value_slow),

    (2, 6, value_fast),
    (2, 3, value_medium)
]


# Logging
state_data = []
log_every = 10

# Play N games
num_epsiodes = 100
for episode_index in range(num_epsiodes):
    print(f"#####\n Episode: {episode_index} \n#####")

    # Choose a reasonable depth and breadth
    depth, breadth, value_fn = config_choices[np.random.randint(len(config_choices))]
    print(depth, breadth, value_fn.__qualname__)

    # Instantiate game
    game = new_game()

    for move_index in tqdm.tqdm(range(2000)):
        if game_over(game):
            df = pd.DataFrame(state_data)
            df.to_csv(f"/Users/alutes/Documents/Data/state_data_{episode_index}.csv")
            break

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
            lookahead_value = score_move_space(move_space[action_name])
            action_values[action_name] = lookahead_value

        # Log Data
        if move_index%log_every == 0:
            state_data.append({
                'episode' : episode_index,
                'move' : move_index,
                'depth' : depth,
                'breadth' : breadth,
                'lookahead_value' : lookahead_value,
                'value_fn' : value_fn.__qualname__,
                'game' : game
            })

        # Choose Action
        move = choose_max(action_values)
        game, done = all_moves_dict[move](game)
        if not done:
            raise ValueError("Illegal move chosen")
        game = add_block(game)