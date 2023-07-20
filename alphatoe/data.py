import math

import torch as t
from torch.nn import functional as F
from torch import Tensor
from tqdm import tqdm

from alphatoe.game import (
    Board,
    next_minimax_moves,
    apply_best_moves,
    generate_all_games,
    get_all_minimax_games,
    next_possible_moves,
)

from typing import Optional


def gen_games(gametype: str = "all"):
    board = [Board()]

    if gametype == "all":
        print("Generating all possible games...")
        games = generate_all_games(board)

    elif gametype == "strat":
        print("Generating strategic games...")
        games = apply_best_moves(board)

    elif gametype == "minimax first":
        print("Generating minimax vs all...")
        games = get_all_minimax_games(board, True, None)

    elif gametype == "minimax second":
        print("Generating minimax vs all...")
        games = get_all_minimax_games(board, False, None)
    else:
        raise ValueError(
            f"gametype must be one of 'all', 'strat', or 'all minimax'. Not {gametype}"
        )

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

def gen_data_prob_encoded_labels(moves: Tensor) -> tuple[Tensor, Tensor]:
    data = moves[:, :-1]
    labels = []
    master_cool_guys: dict[str, list[int]] = {}
    for game in tqdm(data):
        label = []
        for idx in range(1, len(game) + 1):
            # convert tensor to list
            seq: list[int] = game[:idx].tolist()
            seq_str: str = str(seq)
            if seq_str not in master_cool_guys:
                master_cool_guys[seq_str] = next_possible_moves(seq)
            next_moves = master_cool_guys[seq_str]
            encoded_label = [0] * 10
            for i in range(len(encoded_label)):
                if i in next_moves:
                    encoded_label[i] = 1
            encoded_label = t.tensor(encoded_label) / sum(encoded_label)
            label.append(encoded_label)
        labels.append(t.stack(label))
    labels = t.stack(labels)

    print(labels.shape)
    print("Generated all data and probabilistic labels")
    return data, labels


def gen_data_minimax_encoded_labels(moves: Tensor) -> tuple[Tensor, Tensor]:
    data = moves[:, :-1]
    labels = []
    master_cool_guys: dict[str, list[int]] = {}
    for game in tqdm(data):
        label = []
        for idx in range(1, len(game) + 1):
            # convert tensor to list
            seq: list[int] = game[:idx].tolist()
            seq_str: str = str(seq)
            if seq_str not in master_cool_guys:
                master_cool_guys[seq_str] = next_minimax_moves(seq)
            next_moves = master_cool_guys[seq_str]
            encoded_label = [0] * 10
            for i in range(len(encoded_label)):
                if i in next_moves:
                    encoded_label[i] = 1
            encoded_label = t.tensor(encoded_label) / sum(encoded_label)
            label.append(encoded_label)
        labels.append(t.stack(label))
    labels = t.stack(labels)

    print(labels.shape)
    print("Generated all data and minimax labels")
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
        print(labels.shape)
        labels = labels.to(device)
        print(labels.shape)

    return (data[train_inds], labels[train_inds], data[test_inds], labels[test_inds])


def gen_data(
    gametype: str,
    split_ratio: float = 0.8,
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    if gametype == "minimax all":
        _, moves = gen_games("all")
        data, encoded_labels = gen_data_minimax_encoded_labels(moves)
    if gametype == "prob all":
        _, moves = gen_games("all")
        data, encoded_labels = gen_data_prob_encoded_labels(moves)
    else:
        _, moves = gen_games(gametype)
        data, labels = gen_data_labels(moves)
        encoded_labels = gen_data_labels_one_hot(labels)
    return train_test_split(
        data, encoded_labels, split_ratio=split_ratio, device=device, seed=seed
    )
