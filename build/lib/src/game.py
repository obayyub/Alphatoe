import random
import math
from typing import List, Tuple, Optional
from enum import Enum
from collections import defaultdict
from copy import deepcopy

class State(Enum):
    DRAW = 0
    OVER = 1
    ONGOING = 2

class Board():
    def __init__(self):
        self.grid = [' '] * 9 # The game board
        self.turn = 'X' #Who's turn is it?
        self.game_state = State.ONGOING 
        self.winner = '' 
        self.moves_played = [] 
        self.is_maximizer = True # Useful for the minimax algorithm

    # Internal
    def swap_turn(self) -> None:
        if self.turn == 'X':
            self.turn = 'O'
            self.maximizer = not self.is_maximizer
        elif self.turn == 'O':
            self.turn = 'X'
            self.maximizer = not self.is_maximizer
        else:
            raise ValueError(f"Oh god how did we end up here, self.turn is {self.turn}")

    # External
    def get_possible_moves(self) -> List[int] :
        if self.game_state != State.ONGOING:
            return []
        else:
            return [i for i in range(9) if self.grid[i] == ' ']

    # External
    def make_move(self, move: int):
        if move not in self.get_possible_moves():
            raise ValueError("Not a valid move nerd!!")
        self.grid[move] = self.turn
        self.moves_played.append(move)
        self.game_state = self.get_game_state()
        self.swap_turn()
        return self

    # Internal
    def get_game_state(self) -> State:
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), 
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        for condition in win_conditions:
            if self.grid[condition[0]] == self.grid[condition[1]] == self.grid[condition[2]] != ' ':
                self.winner = self.grid[condition[0]]
                return State.OVER
        if ' ' not in self.grid:
            return State.DRAW
        else:
            return State.ONGOING

    # External
    def get_winner(self) -> str:
        return self.winner 

    # External
    def undo(self) -> None:
        last_move = self.moves_played.pop()
        self.grid[last_move] = ' '
        self.swap_turn()

    # External
    # Function to draw the game board
    def draw_board(self):
        row1 = "| {} | {} | {} |".format(self.grid[0], self.grid[1], self.grid[2])
        row2 = "| {} | {} | {} |".format(self.grid[3], self.grid[4], self.grid[5])
        row3 = "| {} | {} | {} |".format(self.grid[6], self.grid[7], self.grid[8])
        print(row1)
        print(row2)
        print(row3)

def generate_all_games(boards: List[Board], finished_boards: List[Board] = []) -> List[Board]:
    ongoing_boards = []
    for board in boards:
        possible_moves = board.get_possible_moves()
        if possible_moves != []:
            for move in possible_moves:
                _board = deepcopy(board)
                ongoing_boards.append(_board.make_move(move))
        else:
            finished_boards.append(board)
    
    if ongoing_boards == []:
        return finished_boards
    else:
        generate_all_games(ongoing_boards, finished_boards=finished_boards)


def apply_best_moves(boards: List[Board]) -> List[Board]: # type: ignore
    game_len = len(boards[0].moves_played) + 1
    print(f"we're on the {game_len}th loop!")
    new_boards = []
    for board in boards:
        for move in get_best_moves(board):
            _board = deepcopy(board)
            new_boards.append(_board.make_move(move))
    print(f"we've got {len(new_boards)} boards, y'all!")

    # boards : List[Board] = [board.make_move(move) for board in boards for move in get_best_moves(board)]
    if new_boards[0].game_state == State.DRAW:
        print(f"we've got {len(new_boards)} boards, y'all!")
        # TODO: Make sure that everything in here is of the same size
        # TODO: Make sure that there are no duplicates
        return new_boards 
    else:
        apply_best_moves(new_boards)
    

def get_best_moves(board) -> List[int]:
    if board.is_maximizer:
        bestScore = -math.inf
    else:
        bestScore = math.inf

    bestMove = []
    for move in board.get_possible_moves():
        board.make_move(move)
        score = minimax(board)
        board.undo()
        if board.is_maximizer & (score >= bestScore):
            bestScore = score
            bestMove.append((move, score))
        elif (not board.is_maximizer) & (score <= bestScore):
            bestScore = score
            bestMove.append((move, score))
    assert bestMove is not []
    bestMove = [move for move, score in bestMove if score == bestScore ]
    return bestMove
    
def make_random_best_move(board) -> None:
    if board.is_maximizer:
        bestScore = -math.inf
    else:
        bestScore = math.inf

    bestMove = []
    for move in board.get_possible_moves():
        board.make_move(move)
        score = minimax(board)
        board.undo()
        if board.is_maximizer & (score >= bestScore):
            bestScore = score
            bestMove.append((move, score))
        elif (not board.is_maximizer) & (score <= bestScore):
            bestScore = score
            bestMove.append((move, score))
    assert bestMove is not []
    bestMove = [move for move, score in bestMove if score == bestScore ]
    bestMove = random.choice(bestMove)
    board.make_move(bestMove)

def minimax(board: Board) -> int:
    if (board.game_state is State.DRAW):
        return 0
    elif (board.game_state is State.OVER):
        return 1 if board.get_winner() is board.is_maximizer else -1

    scores = []
    for move in board.get_possible_moves():
        board.make_move(move)
        scores.append(minimax(board))
        board.undo()

    return max(scores) if board.is_maximizer else min(scores)

# def play_game():
#     board = Board()
#     while board.game_state == State.ONGOING:
#         make_random_best_move(board)
#     return board

