from torch import Tensor
from transformer_lens import HookedTransformer
import torch as t
import tqdm
from alphatoe.game import Board, State, get_best_moves
from collections import Counter
import random


def _sample_game(
    model: HookedTransformer, temp: float, probabilistic=False
) -> list[int]:
    assert temp > 0
    seq = [10]
    # no grad
    with t.no_grad():
        # sample 9 moves plus one end game token
        for _ in range(10):
            logits: Tensor = model(t.tensor(seq))[0, -1]
            probs = t.softmax(logits / temp, dim=0)
            token = int(t.multinomial(probs, num_samples=1).item())
            seq.append(token)
    return seq


def sample_games(
    model: HookedTransformer, temp: float, num_games: int
) -> list[list[int]]:
    games: list[list[int]] = []
    for _ in tqdm.tqdm(range(num_games)):
        games.append(_sample_game(model, temp))
    return games


# evals return True on model error
def _check_played_repeat_moves(game: list[int]) -> bool:
    clean_game = [token for token in game if token != 9]
    set_length = len(set(clean_game))
    return set_length != len(clean_game)


def _check_if_illegal_moves(game: list[int]) -> bool:
    board = Board()
    for move in game[1:-1]:
        if board.game_state == State.ONGOING:
            try:
                board.make_move(move)
            except:
                return True
        elif move == 9:
            pass
        else:
            return True
    return False


def inappropriate_end_state(game: list[int]) -> bool:
    board = Board()
    for move in game[1:]:
        if board.game_state == State.ONGOING and move == 9:
            return True
        try:
            board.make_move(move)
        except:
            return False
    return False


def _check_played_after_player_victory(game: list[int]) -> bool:
    board = Board()
    for move in game[1:]:
        if board.game_state == State.OVER and move != 9:
            return True
        try:
            board.make_move(move)
        except:
            return False
    return False


def _check_played_after_draw_game(game: list[int]) -> bool:
    board = Board()
    for move in game[1:]:
        if board.game_state == State.DRAW and move != 9:
            return True
        try:
            board.make_move(move)
        except:
            return False
    return False


def _check_illegal_moves_again(games: list[list[int]]) -> list[bool]:
    return [
        _check_played_repeat_moves(game)
        or _check_played_after_player_victory(game)
        or _check_played_after_draw_game(game)
        for game in games
    ]


# returns winner
def model_vs_minimax(model: HookedTransformer, minimax_turn: bool) -> str:
    board = Board()

    while board.game_state == State.ONGOING:
        if minimax_turn:
            move = random.choice(get_best_moves(board))
            board.make_move(move)
            minimax_turn = not minimax_turn
        else:
            move = t.argmax(model(t.tensor([10] + board.moves_played))[0, -1]).item()
            board.make_move(move)
            minimax_turn = not minimax_turn
    if board.game_state == State.DRAW:
        return "draw"
    elif minimax_turn:
        return "model"
    else:
        return "minimax"


def _check_minimax_win_rate(model: HookedTransformer, samples: int) -> Counter[str]:
    games = [
        model_vs_minimax(model, i < (samples // 2)) for i in tqdm.tqdm(range(samples))
    ]
    c = Counter(games)
    sm = sum(c.values())
    new_c = {k: c[k] / sm for k in c}
    return new_c


def get_error_rate(games: list[list[int]]) -> float:
    return _check_illegal_moves_again(games).count(True) / len(games)


def eval_model(
    games: list[list[int]], game_evals: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[int, dict[str, bool]]]:
    eval_fs = [
        _check_played_repeat_moves,
        _check_played_after_player_victory,
        _check_played_after_draw_game,
        inappropriate_end_state,
        _check_if_illegal_moves,
    ]
    eval_counts = {func.__name__: 0.0 for func in eval_fs}
    eval_games = {
        i: {func.__name__: False for func in eval_fs} for i in range(len(games))
    }
    game_count = len(games)
    for i, game in tqdm.tqdm(enumerate(games)):
        for func in eval_fs:
            if func(game):
                eval_games[i][func.__name__] = True
                eval_counts[func.__name__] += 1
            else:
                eval_games[i][func.__name__] = False

    for func in eval_fs:
        eval_counts[func.__name__] /= game_count

    if game_evals:
        return eval_counts, eval_games
    else:
        return eval_counts
