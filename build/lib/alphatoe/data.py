import math

import torch as t
from torch.nn import functional as F
from torch import Tensor

from alphatoe.game import Board, apply_best_moves, generate_all_games

from typing import Optional


def gen_games(gametype: str = "all"):
    board = [Board()]

    if gametype == "all":
        print("Generating all possible games...")
        games = generate_all_games(board)

    elif gametype == "strat":
        print("Generating strategic games...")
        games = apply_best_moves(board)

    else:
        raise ValueError(f"gametype must be one of 'all' or 'strat', not {gametype}")

    print(f"Generated {len(games)} games")

    moves = t.tensor(
        [
            [10] + game.moves_played + ([9] * (10 - len(game.moves_played)))
            for game in games
        ],
        requires_grad=False,
    )

    print("Generated array of moves")

    return games, moves


def gen_data_labels(moves: Tensor) -> tuple[Tensor, Tensor]:
    data = moves[:, :-1]
    labels = moves[:, 1:]
    print(labels.shape)
    print("Generated data and labels")
    return data, labels


def gen_data_labels_one_hot(labels: Tensor) -> Tensor:
    encoded_labels: Tensor = F.one_hot(labels).to(t.float)
    print("One hot encoded labels")
    return encoded_labels


def train_test_split(
    data: Tensor,
    labels: Tensor,
    split_ratio: float = 0.8,
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        t.random.manual_seed(seed)

    assert len(data) == len(labels), "data and label lengths don't match"
    assert isinstance(split_ratio, float)
    assert 0 <= split_ratio <= 1

    inds = t.randperm(len(data))
    split = math.floor(split_ratio * len(data))
    train_inds, test_inds = inds[:split], inds[split:]

    if device is not None:
        data = data.to(device)
        labels = labels.to(device)

    return (data[train_inds], labels[train_inds], data[test_inds], labels[test_inds])


def gen_data(
    gametype: str,
    split_ratio: float = 0.8,
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    _, moves = gen_games(gametype)
    data, labels = gen_data_labels(moves)
    encoded_labels = gen_data_labels_one_hot(labels)
    return train_test_split(
        data, encoded_labels, split_ratio=split_ratio, device=device, seed=seed
    )
