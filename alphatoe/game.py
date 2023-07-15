import random
import math
from typing import Optional, Callable
from enum import Enum
from copy import deepcopy
import numpy as np
from alphatoe.trie import Trie
import torch
from torch import Tensor


class State(Enum):
    DRAW = 0
    OVER = 1
    ONGOING = 2


class Board:
    def __init__(self) -> None:
        self.grid = [" "] * 9  # The game board
        self.turn = "X"  # Who's turn is it?
        self.game_state: State = State.ONGOING
        self.moves_played: list[int] = []
        self.is_maximizer = True  # Useful for the minimax algorithm
        self.children: list[Board] = []
        self.winner = ""

    # Internal
    def swap_turn(self) -> None:
        match self.turn:
            case "X":
                self.turn = "O"
                self.is_maximizer = not self.is_maximizer
            case "O":
                self.turn = "X"
                self.is_maximizer = not self.is_maximizer
            case _:
                raise ValueError(
                    f"Oh god how did we end up here, self.turn is {self.turn}"
                )

    def is_over(self) -> bool:
        if self.game_state != State.ONGOING:
            return True
        else:
            return False

    def get_state(self) -> State:
        return self.game_state

    # External
    def get_possible_moves(self) -> list[int]:
        if self.game_state != State.ONGOING:
            return []
        else:
            return [i for i in range(9) if self.grid[i] == " "]

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
        win_conditions = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for condition in win_conditions:
            if (
                self.grid[condition[0]]
                == self.grid[condition[1]]
                == self.grid[condition[2]]
                != " "
            ):
                self.winner = self.grid[condition[0]]
                return State.OVER
        if " " not in self.grid:
            return State.DRAW
        else:
            return State.ONGOING

    # External
    def get_winner(self) -> str:
        if self.winner != "":
            return self.winner
        else:
            raise ValueError("Game's not over yet or it's a draw!")

    # External
    def undo(self) -> None:
        last_move = self.moves_played.pop()
        self.grid[last_move] = " "
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


def get_possible_moves(board: Board) -> list[int]:
    if board.game_state != State.ONGOING:
        return []
    else:
        return [i for i in range(9) if board.grid[i] == " "]


def generate_all_games(
    boards: list[Board], finished_boards: Optional[list[Board]] = None
) -> list[Board]:
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
    if board.game_state == State.DRAW:
        return 0
    elif board.game_state == State.OVER:
        return 10 if board.turn == "O" else -10

    scores: list[int] = []
    for move in board.get_possible_moves():
        board.make_move(move)
        scores.append(minimax(board))
        board.undo()

    return max(scores) if board.is_maximizer else min(scores)


def get_best_moves(board: Board) -> list[int]:
    if board.is_maximizer:
        bestScore = -math.inf
    else:
        bestScore = math.inf

    bestMove: list[tuple[int, int]] = []
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
    bestMoves = [move for move, score in bestMove if score == bestScore]
    return bestMoves


def apply_best_moves(
    boards: list[Board], finished_boards: Optional[list[Board]] = None
) -> list[Board]:
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

    moves: list[tuple[int, int]] = []
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
    bestMoves: list[int] = [move for move, score in moves if score == bestScore]
    bestMove = random.choice(bestMoves)
    board.make_move(bestMove)


def get_all_minimax_games(
    boards: list[Board],
    minimax_turn: bool,
    finished_boards: Optional[list[Board]] = None,
) -> list[Board]:
    if finished_boards == None:
        finished_boards = []
    ongoing_boards = []
    if minimax_turn:
        move_getter = get_best_moves
    else:
        move_getter = get_possible_moves
    for board in boards:
        if board.game_state != State.ONGOING:
            finished_boards.append(board)
        else:
            for move in move_getter(board):
                _board = deepcopy(board)
                _board.make_move(move)
                ongoing_boards.append(_board)
    if ongoing_boards == []:
        return finished_boards
    else:
        return get_all_minimax_games(ongoing_boards, not minimax_turn, finished_boards)


def determine_entropy(counts: list[int]) -> float:
    total_counts = sum(counts)
    entropy = 0.0
    for count in counts:
        entropy += -(count / total_counts) * np.log2(count / total_counts)
    return entropy


# goes over number of unique subsequences of tic tac toe
def generate_tree(board: Board, move_search: Callable[..., list[int]]) -> None:
    if board.is_over():
        return None
    children = []
    for move in move_search(board):
        _board = deepcopy(board)
        _board.make_move(move)
        children.append(_board)
        generate_tree(_board, move_search)
    board.children = children


# goes over number of unique subsequences of tic tac toe
def tree_walk(board: Board, move_search: Callable[..., list[int]]) -> tuple[float, int]:
    if board.is_over():
        return (0, 10 - len(board.moves_played))
    counts = []
    seen = 1
    total_entropy = 0.0
    for move in move_search(board):
        board.make_move(move)
        extra_entropy, extra_seen = tree_walk(board, move_search)
        counts.append(extra_seen)
        seen += extra_seen
        total_entropy += extra_entropy
        board.undo()
    local_entropy = determine_entropy(counts)
    return (total_entropy + local_entropy, seen)


def entropy(subseq: list[int], trie: Trie) -> float:
    node = trie.get_sub_node(subseq)
    counts = [child.game_count for child in node]
    # sum of array
    total_count = sum(counts)
    # drop 0 counts
    counts = [count for count in counts if count != 0]
    entropy = [
        (-count / total_count) * math.log(count / total_count) for count in counts
    ]
    return sum(entropy)


def avg_entropy(data_set: list[list[int]], trie: Trie):
    total_entropy = 0
    for sample in data_set:
        for i in range(len(sample)):
            total_entropy += entropy(sample[:i], trie)

    return total_entropy / (10 * len(data_set))


"""
coin flips
1. p(H) * log2(p(H)) + p(T) * log2(p(T))
2. p(H | H) * log2(p(H | H)) + p(T | H) * log2(p(T | H)) + p(H | T) * log2(p(H | T)) + p(T | T) * log2(p(T | T)) 

"""


def probabilistic_guess(seq: Tensor):
    if seq[-1] == 9:
        out = torch.zeros(10)
        out[-1] = 1
        return out
    moves_left = 10 - len(seq)
    logits = torch.zeros(10) - torch.inf
    if moves_left > 0:
        move_prob = 1 / moves_left
    elif moves_left == 0:
        logits[-1] = 1
        return logits
    else:
        raise ValueError(f"Length of sequence must be less than or equal to 10")
    for move in range(9):
        if move not in seq:
            logits[move] = move_prob
    assert sum(torch.softmax(logits, dim=0)) == 1
    return logits


def autoregressive_guess(seq: list[int]):
    logits = []
    for i in range(1, len(seq) + 1):
        logits.append(probabilistic_guess(seq[:i]))

    # The unsqueeze is just to make our evals happy
    return torch.stack(logits).unsqueeze(dim=0)


def play_game(seq: list[int]):
    board = Board()
    for move in seq[1:-1]:
        try:
            board.make_move(move)
        except:
            print("Invalid game")
            break
    board.draw_board()
