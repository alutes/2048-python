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


def softmax(p, temperature = .05):
    expp = np.exp(p / temperature)
    return expp / np.sum(expp)

# Choose a random value among weighted options
def choose_one_random(prob_dict, use_softmax = True):
    choices = list(prob_dict.keys())
    scores = np.array(list(prob_dict.values()))
    
    if use_softmax:
        scores = softmax(scores)
    
    return random.choices(
        choices,
        weights=scores,
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

all_moves = [up, down, left, right]
all_moves_dict = {
    'up' : up,
    'down' : down,
    'left' : left,
    'right' : right    
    } 