#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:18:30 2023

@author: alutes
"""
import random
import constants as c
import numpy as np
import copy
from logic import all_moves, add_block, game_over, choose_one_random
from value_functions import VALUE_MODEL, value_slow, weighted_average, value_medium

# Exploration Constants
GAMMA = 0.95
LOSS_REWARD = -100

########################################
##
##  Search game states out to a depth of N
##
########################################

# Build a dictionary of possible actions and their resulting game states
def build_move_dict(game, possible_moves = all_moves):
    move_states = {}
    for move in possible_moves:
        move_state, done = move(game)

        if done:
            move_states[move.__qualname__] = move_state

    return move_states

# Choose n random values among weighted options
def choose_n_random(prob_dict, k):
    choices = list(prob_dict.keys())
    scores = np.array(list(prob_dict.values()))

    return random.choices(
        choices,
        weights=scores,
        k=k
          )

# Choose a random value among weighted options
def choose_max(prob_dict):
    return max([(value, key) for key, value in prob_dict.items()])[1]

# Build a dictionary of possible actions block generation steps and resulting boards
def build_gen_dict(game, max_gen_states):

    # Find all possible empty spots and value combinations
    empties = {}
    for i in range(game.shape[0]):
        for j in range(game.shape[1]):
            if game[i,j] == 0:
                # If we find an empty value then try adding a 2 or 4
                empties[(i,j,2)] = get_board_probability((i,j,2))
                empties[(i,j,4)] = get_board_probability((i,j,4))

    # If there are more than max_gen_states possibilities, trim them
    if len(empties) > max_gen_states:
        empties = choose_n_random(empties, max_gen_states)

    # If we find an empty value then try adding a 2 or 4
    board_states = {}
    for i,j,new_value in empties:
        new_state = copy.deepcopy(game)
        new_state[i,j] = new_value

        # Store in a dict with indices: row, column, new_value
        board_states[(i,j,new_value)] = new_state

    return board_states


# Expand out both moves and generation
def expand_and_prune(
        game,
        depth = 2,
        value_fn = value_medium,
        max_initial_states = 8,
        max_retained_actions = 2,
        max_retained_states = 1
        ):

        # Expand all possibilities at depth of 1
        result = expand_states(game, max_initial_states)

        # Prune possibile actions
        result = prune_action_states(
            expanded_state_dict = result,
            value_fn =value_fn,
            max_retained_actions = max_retained_actions,
            max_retained_states = max_retained_states
            )

        # If we have remaining depth to go, recursively continue to expand each branch
        if depth > 1:
            for action in result:
                for state in result[action]:
                    result[action][state] = expand_and_prune(
                            game = result[action][state],
                            depth = depth - 1,
                            value_fn =value_fn,
                            max_initial_states = max_initial_states,
                            max_retained_actions = max_retained_actions,
                            max_retained_states = max_retained_states
                        )

        return result

# Expand out both moves and generation
def expand_states(
        game,
        max_initial_states = 8
        ):

    # All possible game states from each valid action
    move_dict = build_move_dict(game)

    # States of each possible generated state after each action
    expanded_state_dict = {}

    # Expand Game states based on where the new block could be generated
    for action in move_dict:

        # All possible game states from block generation
        action_dict = build_gen_dict(
            move_dict[action],
            max_gen_states = max_initial_states
            )
        expanded_state_dict[action] = action_dict

    return expanded_state_dict


# Prune Actions to the top N based on highest expected value
def prune_action_states(
        expanded_state_dict,
        value_fn,
        max_retained_actions = 2,
        max_retained_states = 1
        ):

    # Prune actions based on value at depth = 1
    chosen_actions = choose_best_actions(
        expanded_state_dict = expanded_state_dict,
        value_fn = value_fn,
        max_retained_actions = max_retained_actions
     )

    # Keep only the top N actions
    new_state_dict = {}
    for action in chosen_actions:

        # Otherwise, Prune block generation states among remaining actions, if there are more than max_retained_states
        new_state_dict[action] = prune_gen_states(
            action_dict = expanded_state_dict[action],
            max_retained_states = max_retained_states
            )
    return new_state_dict


# Find the best actions according to a value function
def choose_best_actions(
        expanded_state_dict,
        value_fn,
        max_retained_actions
        ):
    
    action_values = {}
    
    for action in expanded_state_dict:

        # Calculate Expected Board Value
        action_values[action] = score_move_space(expanded_state_dict[action], value_fn, discount = 1)


    # Find the Nth highest action value
    actions_sorted_by_value = sorted(
        list(action_values.keys()), 
        key = action_values.get, 
        reverse = True
        )
    
    return actions_sorted_by_value[0:max_retained_actions]

# Prune Generated blocks to the top N, randomly
def prune_gen_states(
        action_dict,
        max_retained_states = 1
        ):
    if len(action_dict) > max_retained_states:
        state_probabilities = {}

        # Find probabilities of each state
        for state in action_dict:
            state_probabilities[state] = get_board_probability(state)

        # Keep N states
        states_to_keep = choose_n_random(state_probabilities, max_retained_states)

        action_dict = dict([(s, action_dict[s]) for s in states_to_keep])

    return action_dict


# Score a game board using
# S(s) = r(s) + V(s)
# and V is a manually determined board quality function
def score_board(game, value_fn):
    if game_over(game):
        return LOSS_REWARD
    #return value_simple(game)
    return value_fn(game)

# Relative probability of seeing a given generation step
# (based on generation probabilities)
# 90% for a 2
# 10% for a 4
def get_board_probability(gen_key):
    _, _, new_val = gen_key
    return c.GEN_VALUE_PROBS[new_val]

# Backpropogate board scores to the previous move
# - For ACTIONS: take the max (i.e. we would take the best action)
# - For GENERATION: take an expected value since generation is random
def score_move_space(move_space, value_fn, discount = 1):
    # If this is a board then return the board value
    if isinstance(move_space, np.ndarray):
        return discount * score_board(move_space, value_fn = value_fn)

    # Otherwise we should have a more expanded move space
    elif isinstance(move_space, dict):

        # If this is an empty dictionary then we've already lost
        if len(move_space) == 0:
            return LOSS_REWARD

        # Sample the dict key to see what it is
        sample_key = next(iter(move_space.keys()))

        # If the dictionary keys are indices then these are block generation
        # Average over them
        if isinstance(sample_key, tuple):
            return get_expected_value(
                move_space = move_space, 
                value_fn = value_fn,
                discount = discount
                )

        # If the dictionary keys are moves then these are block actions
        # Take the max (off-policy valuation)
        elif isinstance(sample_key, str):
            return np.max(np.array([score_move_space(move_state, value_fn = value_fn, discount = discount * GAMMA,) for move_state in move_space.values()]))

        else:
            raise ValueError(f"{sample_key} is a {type(sample_key)} not valid for move states")

    return ValueError("Move Space not recognized. Should be dict or np.array")

# Expected value from an action
def get_expected_value(move_space, value_fn, discount):
    expected_value = 0.0
    total_probability = 0.0
    for gen_key, move_state in move_space.items():
        probability = get_board_probability(gen_key)
        value = score_move_space(
            move_state, 
            value_fn = value_fn, 
            discount = discount
            )
        expected_value += probability * value
        total_probability += probability
    return expected_value / total_probability


# Value of a state based on the best move available in that state
def get_max_value(move_space, value_fn, discount):
    return np.max(np.array([score_move_space(
        move_state, 
        value_fn = value_fn,
        discount = discount * GAMMA
        ) for move_state in move_space.values()]))


# Depth of a dictionary
def depth(d):
     if isinstance(d, dict):
         return 1 + (max(map(depth, d.values())) if d else 0)
     return 0

# Total Elements in a nested dictionary
def total_length(d):
     if isinstance(d, dict):
         return (sum(map(total_length, d.values())) if d else 0)
     return 1


########################################
##
##  Collapse
##
########################################

def board_probabilities(board_states):
    # Number of possible spaces to add
    board_probabilities = {}
    num_options = len(board_states) / 2

    # Set probabilities for each state
    for index in board_states:
        i,j,new_val = index
        board_probabilities[index] = c.GEN_VALUE_PROBS[new_val] / num_options

    return board_probabilities


########################################
## Monte Carlo Tree Search (MCTS)
##
##  play N random games until you lose or hit depth M.
##  Chose the move that loses the least
##
########################################

def make_simple_move(game):
    move_successful = False

    while not move_successful:
        move = random_move()
        next_game_state, move_successful  = move(game)

    next_game_state = add_block(next_game_state)
    return next_game_state

# Return a value from a simulation which reached depth N and resulted in
# game state
def simulation_value(final_game_state, pct_of_max_depth_reached, ended_in_loss):
    # If we ended in a loss, return a little bit of value for how far we got
    if ended_in_loss:
        return pct_of_max_depth_reached * VALUE_MODEL['depth_achived_weight']

    # If we reached max depth and are still going, give a base value of credit
    # for not losing and then evaluate the board state
    return weighted_average([
        (1.0, VALUE_MODEL['not_loss_weight']), # value of not having lost
        (value_slow(final_game_state), VALUE_MODEL['board_state_weight'])
         ])

# Run a single simulation at the appropriate depth
def mcts_simulate(trial_state, max_simulation_depth = 3):
    ended_in_loss = False

    # We are simulating from right after a move but before a block is added,
    # so must add a block first
    trial_state = add_block(trial_state)

    for move_depth in range(max_simulation_depth):
        if game_over(trial_state):
            ended_in_loss = True
            break
        trial_state = make_simple_move(trial_state)

    return simulation_value(trial_state, move_depth / max_simulation_depth, ended_in_loss)


# Run a single simulation until loss
def mcts_inf_depth(trial_state):
    # We are simulating from right after a move but before a block is added,
    # so must add a block first
    trial_state = add_block(trial_state)
    depth = 0
    while not game_over(trial_state):
        trial_state = make_simple_move(trial_state)
        depth += 1
    return depth


# Run N simulations and return the results
def mcts_run_simulations(game, n_trials = 100, **kwargs):
    trial_results = []
    for trial in range(n_trials):
        trial_state = copy.deepcopy(game)
        trial_results.append(mcts_simulate(trial_state, **kwargs))
    return trial_results


def value_mcts(
        game,
        expansion_depth = 1,
        **kwargs
        ):
    """
    https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#/media/File:MCTS-steps.svg

    Parameters
    ----------
    game : np.array
        the matrix representing the game state.
    expansion_depth : int
        how deep to expand the game tree before simulation/backprop
    max_simulation_depth : TYPE, optional
        DESCRIPTION. The default is 100.
    n_trials : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    float
        DESCRIPTION.

    """
    simulation_results = mcts_run_simulations(game, **kwargs)
    return np.mean(simulation_results)


########################################
## Option III
## DQN
##
## Learn a representation of the board
##  S -> V
##
## Such that
##  V(s) =~ max_a(V(s, a))  via TD Learning
##  V(s) =~ expected score  via MCTS
##
########################################
def value_dqn_score(game):
    # @TODO: Fill in
    return 1.0


####################################
# AI Action choice will go here
####################################

def build_move_states(game):
    # Try each move and build a dict of resulting states
    move_states = {}
    for action in all_moves:
        action_game, action_successful = action(game)

        if action_successful:
            move_states[action] = action_game
    return move_states

def ai(game):

    move_states = build_move_states(game)

    # Pick an action according to the policy
    key = policy(move_states)

    # Apply the move to this game
    return key(game)


####################################
# Policies will go here
####################################

# A probability function of actions for a given game state P(A|S)
# @TODO: Make this informed by quality functions Q(S,A) -> P(A|S)
# Currently a shell which picks randomly over valid moves
def policy(move_states, value_fn = value_slow):
    move_probs = {}

    for action, action_game in move_states.items():
        move_probs[action] = value_fn(action_game)

    return choose_one_random(move_probs, use_softmax = True)


def random_policy(move_states):
    return np.choose(1, list(move_states.keys()))


def random_move(moves = all_moves):
    return np.random.choice(moves, 1).item()
