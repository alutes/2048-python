import random
import constants as c
import numpy as np
import copy

##########
# Basic Game board
##########

# New Game Matrix
def new_game():
    matrix = np.zeros([c.nrow, c.ncol]).astype(int)
    for i in range(c.NUM_START_BLOCKS):
        matrix = add_block(matrix)
    return matrix

# Add a block to a random free space
def add_block(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a][b] = choose_new_block_value()
    return mat

# Choose a random value among weighted options
def choose_one_random(probs):
    return random.choices(
        list(probs.keys()),
        weights=probs.values(),
        k=1
          )[0]

# Choose a random value for a new block
def choose_new_block_value():
    return choose_one_random(c.GEN_VALUE_PROBS)


def num_free_spaces(mat):
    return np.equal(mat, 0).sum()

# Check if the game is over
def game_over(mat):

    # check for any zero entries
    if num_free_spaces(mat) > 0:
        return False

    # check for same cells that touch each other
    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        # more elegant to use exceptions but most likely this will be their solution
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return False

    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return False
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return False
    return True

def reverse(mat):
    return np.flip(mat)

def transpose(mat):
    return np.transpose(mat)


##########
# Valid Game Moves
##########

def cover_up(mat):
    new = np.zeros(list(mat.shape))
    done = False
    for i in range(mat.shape[0]):
        count = 0
        for j in range(mat.shape[1]):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done

def merge(mat, done):
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] += 1
                mat[i][j+1] = 0
                done = True
    return mat, done

def up(game):
    # return matrix after shifting up
    game = transpose(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done

def down(game):
    # return matrix after shifting down
    game = reverse(transpose(game))
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return game, done

def left(game):
    # return matrix after shifting left
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    return game, done

def right(game):
    # return matrix after shifting right
    game = reverse(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = reverse(game)
    return game, done

all_moves = [up, down, left, right]

####################################
# AI Action choice will go here
####################################

def ai(game):
    
    # Try each move and build a dict of resulting states
    move_states = {}
    for action in all_moves:
        action_game, action_successful = action(game)
        
        # Don't attempt this move if it will end the game
        if action_successful:
            move_states[action] = action_game
    
    # Pick an action according to the policy
    key = policy(move_states)
    
    # Apply the move to this game
    return key(game)
    

####################################
# Policies will go here
####################################

def softmax(p, temperature = .05):
    expp = np.exp(p / temperature)
    return expp / np.sum(expp)

# A probability function of actions for a given game state P(A|S)
# @TODO: Make this informed by quality functions Q(S,A) -> P(A|S)
# Currently a shell which picks randomly over valid moves
def policy(move_states):
    move_probs = {}
    
    for action, action_game in move_states.items():
        move_probs[action] = value_function(action_game)
        
    # Softmax
    
    return choose_one_random(move_probs)


def random_policy(move_states):        
    return np.choose(1, list(move_states.keys()))


def random_move(moves = all_moves):        
    return np.random.choice(moves, 1).item()


####################################
# Value Functions will go here
####################################

# A fitness function for any particular game state
# @TODO: Make this informed by the different types of value functions below
def value_function(game):
    return value_mcts(game)


########################################
## Option I 
## Custom Formula for fitness function which incorporates 3 
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
########################################

VALUE_MODEL = {
   'smooth_weight' : 1.0,           # relative weight of smoothness
   'monotonicity_weight' : 0.0,     # relative weight of monotonicity
   'free_weight' : 1.0,             # relative weight of free spaces
   'board_state_weight' : 1.0,       # relative value of the board state compared to not losing
   'not_loss_weight' : 3.0,         # relative weight of not losing after max play depth
   'depth_achived_weight' : 1.0     # relative weight of achieving % of max play depth before losing (must be less than not_loss_weight)
    }

# % of spaces which are free
#  mean_ij(f_ij = 0)
def free_spaces(game):
    return num_free_spaces(game) / game.size

# 0-1 measure of smoothness
# 1 - |∇f| / |f|
# |∇f| = |g([∂f/∂x, ∂f/∂y])| 
def smoothness(game):
    return 1 - np.linalg.norm(np.gradient(game)) / (2 * np.linalg.norm(game))

# 0-1 measure of monotonicity
def monotonicity(game):
# @TODO: Fill in
    return 1.0

def weighted_average(value_weight_pairs):
    total = 0.0
    total_weight = 0.0
    for value, weight in value_weight_pairs:
        total += value * weight
        total_weight += weight
    return total / total_weight

def value_manual_score(
        game, 
        monotonicity_weight = VALUE_MODEL['monotonicity_weight'], 
        smooth_weight = VALUE_MODEL['smooth_weight'], 
        free_weight = VALUE_MODEL['free_weight']
        ):
    return weighted_average([
        (monotonicity(game), monotonicity_weight), 
        (smoothness(game), smooth_weight), 
        (free_spaces(game), free_weight)
        ])

########################################
## Option II
## Monte Carlo Tree Search (MCTS)
##
##  play N random games until you lose or hit depth M. 
##  Chose the move that loses the least
##
########################################

def make_random_move(game):
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
        (value_manual_score(final_game_state), VALUE_MODEL['board_state_weight'])
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
        trial_state = make_random_move(trial_state)
        
    return simulation_value(trial_state, move_depth / max_simulation_depth, ended_in_loss)


# Run a single simulation until loss
def mcts_inf_depth(trial_state):
    # We are simulating from right after a move but before a block is added,
    # so must add a block first
    trial_state = add_block(trial_state)
    depth = 0
    while not game_over(trial_state):
        trial_state = make_random_move(trial_state)
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
