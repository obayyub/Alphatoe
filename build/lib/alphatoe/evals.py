from torch import Tensor
from transformer_lens import HookedTransformer
import torch as t
import tqdm
from alphatoe.game import Board


def _sample_game(
    model: HookedTransformer, temp: float, probabilistic=False
) -> list[int]:
    assert temp > 0
    seq = [10]
    # no grad
    with t.no_grad():
        for _ in range(8):
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


# evals return True on model error
def _check_played_repeat_moves(game: list[int]) -> bool:
    clean_game = [token for token in game if token != 9]
    set_length = len(set(clean_game))
    return set_length != len(clean_game)


def _check_played_after_finished_game(game: list[int]) -> bool:
    board = Board()
    for move in game[1:]:
        try:
            board.make_move(move)
        except:
            return True
    return False


def _check_illegal_moves_again(games: list[list[int]]) -> list[bool]:
    return [
        _check_played_repeat_moves(game) or _check_played_after_finished_game(game)
        for game in games
    ]


def error_rate(games: list[list[int]]) -> float:
    return _check_illegal_moves_again(games).count(True) / len(games)
