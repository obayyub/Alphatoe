import random
import math
from typing import List, Tuple, Optional
from enum import Enum
from copy import deepcopy

class State(Enum):
    DRAW = 0
    OVER = 1
    ONGOING = 2

class Board():
    def __init__(self):
        self.grid = [' '] * 9 # The game board
        self.turn = 'X' #Who's turn is it?
        self.game_state: State = State.ONGOING 
        self.moves_played: list[int] = [] 
        self.is_maximizer = True # Useful for the minimax algorithm

    # Internal
    def swap_turn(self) -> None:
        match self.turn:
            case 'X':
                self.turn = 'O'
                self.is_maximizer = not self.is_maximizer
            case 'O':
                self.turn = 'X'
                self.is_maximizer = not self.is_maximizer
            case _:
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
            raise ValueError(f"{move} is not a valid move nerd!!")
        self.grid[move] = self.turn
        self.moves_played.append(move)
        self.game_state = self.set_game_state()
        self.swap_turn()
        return self

    # Internal
    def set_game_state(self) -> State:
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
        if self.winner != None:
            return self.winner
        else:
            raise ValueError("Game's not over yet or it's a draw!")


    # External
    def undo(self) -> None:
        last_move = self.moves_played.pop()
        self.grid[last_move] = ' '
        self.game_state = State.ONGOING
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

def generate_all_games(boards: List[Board], finished_boards: Optional[List[Board]] = None) -> List[Board]:
    if finished_boards == None:
        finished_boards = []
    ongoing_boards: list[Board] = []
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
        return generate_all_games(ongoing_boards, finished_boards=finished_boards)

def minimax(board: Board) -> int:
    if (board.game_state == State.DRAW):
        return 0
    elif (board.game_state == State.OVER):
        return 10 if board.turn == "O" else -10 

    scores: list[int]= []
    for move in board.get_possible_moves():
        board.make_move(move)
        scores.append(minimax(board))
        board.undo()

    return max(scores) if board.is_maximizer else min(scores)

def get_best_moves(board: Board) -> List[int]:
    if board.is_maximizer:
        bestScore = -math.inf
    else:
        bestScore = math.inf

    bestMove: List[Tuple[int, int]] = []
    for move in board.get_possible_moves():
        board.make_move(move)
        score = minimax(board)
        board.undo()
        if board.is_maximizer and (score >= bestScore):
            bestScore = score
            bestMove.append((move, score))
        elif (not board.is_maximizer) and (score <= bestScore):
            bestScore = score
            bestMove.append((move, score))
    assert bestMove != []
    bestMoves = [move for move, score in bestMove if score == bestScore ]
    return bestMoves

def apply_best_moves(boards: List[Board], finished_boards: Optional[List[Board]] = None) -> List[Board]:
    if finished_boards == None:
        finished_boards = []
    ongoing_boards = []
    for board in boards:
        if board.game_state != State.ONGOING:
            finished_boards.append(board)
        else:
            for move in get_best_moves(board):
                _board = deepcopy(board)
                _board.make_move(move)
                ongoing_boards.append(_board)
    if ongoing_boards == []:
        return finished_boards
    else:
        return apply_best_moves(ongoing_boards, finished_boards)
    
def make_random_best_move(board: Board) -> None:
    if board.is_maximizer:
        bestScore = -math.inf
    else:
        bestScore = math.inf

    moves: list[tuple[int,int]] = []
    for move in board.get_possible_moves():
        board.make_move(move)
        score = minimax(board)
        board.undo()
        if board.is_maximizer & (score >= bestScore):
            bestScore = score
            moves.append((move, score))
        elif (not board.is_maximizer) & (score <= bestScore):
            bestScore = score
            moves.append((move, score))
    assert moves is not []
    bestMoves: list[int] = [move for move, score in moves if score == bestScore ]
    bestMove = random.choice(bestMoves)
    board.make_move(bestMove)

# def play_game():
#     board = Board()
#     while board.game_state == State.ONGOING:
#         make_random_best_move(board)
#     return board

