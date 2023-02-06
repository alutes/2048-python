#
# CS1010FC --- Programming Methodology
#
# Mission N Solutions
#
# Note that written answers are commented out to allow us to run your
# code easily while grading your problem set.

import random
import constants as c
import numpy as np

#######
# Task 1a #
#######

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 1 mark for creating the correct matrix

def new_game():
    matrix = np.zeros([c.nrow, c.ncol]).astype(int)
    for i in range(c.NUM_START_BLOCKS):
        matrix = add_block(matrix)
    return matrix

###########
# Task 1b #
###########

# [Marking Scheme]
# Points to note:
# Must ensure that it is created on a zero entry
# 1 mark for creating the correct loop

def add_block(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a][b] = choose_new_block_value()
    return mat

def choose_one_random(probs):
    return random.choices(
        list(probs.keys()),
        weights=probs.values(),
        k=1
          )[0]

def choose_new_block_value():
    return choose_one_random(c.GEN_VALUE_PROBS)

###########
# Task 1c #
###########

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 0 marks for completely wrong solutions
# 1 mark for getting only one condition correct
# 2 marks for getting two of the three conditions
# 3 marks for correct checking

def game_over(mat):
    # check for any zero entries
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
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

###########
# Task 2a #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices

def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

###########
# Task 2b #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices

def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new

##########
# Task 3 #
##########

# [Marking Scheme]
# Points to note:
# The way to do movement is compress -> merge -> compress again
# Basically if they can solve one side, and use transpose and reverse correctly they should
# be able to solve the entire thing just by flipping the matrix around
# No idea how to grade this one at the moment. I have it pegged to 8 (which gives you like,
# 2 per up/down/left/right?) But if you get one correct likely to get all correct so...
# Check the down one. Reverse/transpose if ordered wrongly will give you wrong result.

def cover_up(mat):
    new = []
    for j in range(c.GRID_LEN):
        partial_new = []
        for i in range(c.GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
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
                mat[i][j] *= 2
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

####################################
# AI Action choice will go here
####################################
all_moves = [up, down, left, right]

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
    
    
# A probability function of actions for a given game state P(A|S)
# @TODO: Make this informed by quality functions Q(S,A) -> P(A|S)
# Currently a shell which picks randomly over valid moves
def policy(move_states):
    move_probs = {}
    
    for action, action_game in move_states.items():
        move_probs[action] = value_function(action_game)
        
    return choose_one_random(move_probs)


# A fitness function for any particular game state
# @TODO: Make this informed by the different types of value functions below
def value_function(game):
    return value_manual_score(game)


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

def free_spaces(game):
    return np.equal(game, 0).sum()

def smoothness(game):
    return np.linalg.norm(np.gradient(game))

def monotonicity(game):
# @TODO: Fill in
    return 1.0


def value_manual_score(game, monotonicity_weight = 1.0, smooth_weight = 1.0, free_weight = 1.0):
    return monotonicity_weight * monotonicity(game) + \
            smooth_weight * smoothness(game) + \
            free_weight * free_spaces(game)


########################################
## Option II
## Monte Carlo Tree Search (MCTS)
##
##  play N random games until you lose or hit depth M. 
##  Chose the move that loses the least
##
########################################
def value_mcts_score(game):
    # @TODO: Fill in
    return 1.0




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
