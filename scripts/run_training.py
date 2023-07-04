import json
import datetime
import argparse
import os
import re
from datetime import datetime

import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer

from alphatoe import data, train


def main(args: argparse.Namespace):
    check_args(args)
    if args.fine_tune == "recent":
        dir = os.path.dirname(os.path.realpath(__file__))
        model_name = get_most_recent_file(f"{dir}/")
    else:
        model_name = args.model_name

    cfg = HookedTransformerConfig(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_head=args.d_head,
        d_mlp=args.d_mlp,
        act_fn=args.act_fn,
        normalization_type=args.normalization_type,
        d_vocab=11,
        d_vocab_out=10,
        n_ctx=10,
        init_weights=True,
        device=args.device,
        seed=args.seed,
    )

    train_data, train_labels, test_data, test_labels = data.gen_data(
        args.gametype,
        split_ratio=args.train_test_split,
        device=args.device,
        seed=args.seed,
    )
    model = load_model(model_name).to(cfg.device)

    timestamp = make_timestamp()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train.train(
        model,
        train_data,
        train_labels,
        test_data,
        test_labels,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    save_model(model, args.experiment_name, timestamp)
    save_params(args, timestamp)


def check_args(args: argparse.Namespace):
    assert args.d_model % args.d_head == 0


def make_timestamp():
    t = datetime.now()

    return f"{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}"


def save_params(args: argparse.Namespace, timestamp: str) -> None:
    with open(f"{args.experiment_name}-{timestamp}.json", "w+") as f:
        json.dump(vars(args), f)


def save_model(m: HookedTransformer, experiment_name: str, timestamp: str):
    torch.save(m, f"{experiment_name}-{timestamp}.pt")


def load_model(model_name: str) -> HookedTransformer:
    return torch.load(f"{model_name}")


def get_most_recent_file(directory):
    files = os.listdir(directory)

    timestamp_regex = re.compile(r"(\d{8}-\d{6})\.pt")

    files = [f for f in files if timestamp_regex.search(f)]

    if files == []:
        raise ValueError("You need to have run models before you can fine tune them!")

    files.sort(
        key=lambda f: datetime.strptime(
            timestamp_regex.search(f).group(1), "%Y%m%d-%H%M%S"
        ),
        reverse=True,
    )

    most_recent_file = files[0]  # The most recent file
    return most_recent_file


# Get the most recent file in the "scripts" directory


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_name", type=str)
    ap.add_argument("gametype", type=str, choices=["strat", "all"])

    ap.add_argument("--fine_tune", type=str)
    ap.add_argument("--n_epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--train_test_split", type=float, default=0.8)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_head", type=int, default=16)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--act_fn", type=str, default="relu")
    ap.add_argument("--normalization_type", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)

    main(ap.parse_args())
