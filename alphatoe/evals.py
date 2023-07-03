from torch import Tensor
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import numpy as np
import torch as t
import tqdm
#functional
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Any

def _sample_game(model: HookedTransformer, temp: float) -> list[int]:
    assert temp > 0
    seq = [10]
    #no grad
    with t.no_grad():
        for _ in range(8):
            logits = model(t.tensor(seq))[0, -1]
            probs = t.softmax(logits / temp, dim=0)
            token = t.multinomial(probs, num_samples=1).item()
    return seq

def sample_games(model: HookedTransformer, temp: float, num_games: int) -> list[list[int]]:
    games=[]
    for _ in tqdm.tqdm(range(num_games)):
        games.append(_sample_game(model, temp))
    return games

def _check_illegal_moves(game: list[int]) -> bool:
    clean_game = [token for token in game if token != 9]
    set_length = len(set(clean_game))
    return set_length == len(clean_game)

def _check_illegal_moves_again(games: list[list[int]]) -> list[bool]:
    return [_check_illegal_moves(game) for game in games]

def error_rate(games: list[list[int]]) -> float:
    return _check_illegal_moves_again(games).count(False) / len(games)