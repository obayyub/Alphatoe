import math
from torch import Tensor
from transformer_lens import HookedTransformer
import torch as t
import tqdm
from alphatoe.game import Board, State


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
            token: int = t.multinomial(probs, num_samples=1).item()
            seq.append(token)
    return seq


def sample_games(
    model: HookedTransformer, temp: float, num_games: int
) -> list[list[int]]:
    games: list[list[int]] = []
    for _ in tqdm.tqdm(range(num_games)):
        games.append(_sample_game(model, temp))
    return games


def infer_games(
    model: HookedTransformer, game_list: t.Tensor, batch_size: int = 1,
) -> t.Tensor:
    """Running model inference over a list of games."""
    num_games = game_list.shape[0]
    batch_predictions = list()
    for batch_i in tqdm.tqdm(range(math.ceil(num_games / batch_size))):
        start = batch_size * batch_i
        end = min(num_games, start + batch_size)

        input_batch = game_list[start:end, :10]
        logits_batch = model(input_batch)
        batch_predictions.append(t.argmax(logits_batch, axis=-1))

    return t.cat(batch_predictions, axis=0)


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


def get_error_rate(games: list[list[int]]) -> float:
    return _check_illegal_moves_again(games).count(True) / len(games)


def eval_model(games: list[list[int]]) -> dict[str, float]:
    eval_fs = [
        _check_played_repeat_moves,
        _check_played_after_player_victory,
        _check_played_after_draw_game,
        inappropriate_end_state,
        _check_if_illegal_moves,
    ]
    evals = {func.__name__: 0.0 for func in eval_fs}
    game_count = len(games)
    for game in games:
        for func in eval_fs:
            if func(game):
                evals[func.__name__] += 1
            else:
                pass

    for func in eval_fs:
        evals[func.__name__] /= game_count

    return evals
