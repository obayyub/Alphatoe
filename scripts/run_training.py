import json
import datetime
import argparse
import os
import re
from datetime import datetime
import pandas as pd

import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer

from alphatoe import data, train, evals


def main(args: argparse.Namespace):
    print(args)
    print(type(args))
    dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(dir, "models/")
    check_args(args)

    # This is a bit messy, but the logic is clear. If we end up needing more script args we can extract this
    if args.fine_tune == None:
        cfg = create_hooked_transformer_config(args)
        model = HookedTransformer(cfg)
    elif args.fine_tune == "recent":
        model_name = get_most_recent_file(f"{model_dir}/")
        new_args = load_config(model_name, model_dir)
        args = splice_args(args, new_args)
        print(args)
        cfg = create_hooked_transformer_config(args)
        model = HookedTransformer(cfg)
        model.load_state_dict(load_weights(model_name, model_dir))
    else:
        model_name = args.model_name
        args = load_config(model_name, model_dir)
        cfg = create_hooked_transformer_config(args)
        model = HookedTransformer(cfg)
        model.load_state_dict(load_weights(model_name, model_dir))

    model.to(cfg.device)

    train_data, train_labels, test_data, test_labels = data.gen_data(
        args.gametype,
        split_ratio=args.train_test_split,
        device=args.device,
        seed=args.seed,
    )

    timestamp = make_timestamp()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"model checkpoints is: {args.save_checkpoints}")
    print(f"layer count is: {args.n_layers}")
    if args.save_attention_weights:
        model, training_data, attention_weights_data = train.train(
            model,
            train_data,
            train_labels,
            test_data,
            test_labels,
            optimizer=optimizer,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            save_losses=args.save_losses,
            save_checkpoints=args.save_checkpoints,
            save_attention_weights=args.save_attention_weights,
        )
    else:
        model, training_data = train.train(
            model,
            train_data,
            train_labels,
            test_data,
            test_labels,
            optimizer=optimizer,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            save_losses=args.save_losses,
            save_checkpoints=args.save_checkpoints,
            save_attention_weights=args.save_attention_weights,
        )

    print("Saving model weights to disk...")
    save_weights(model, args.experiment_name, timestamp, model_dir)
    print("Model weights saved!")
    save_config(args, timestamp, model_dir)
    if args.save_losses or args.save_checkpoints:
        print("Saving training data to disk...")
        save_training_data(training_data, args.experiment_name, timestamp, model_dir)
        print("Training data saved!")

    if args.save_attention_weights:
        print("Saving attention weights to disk...")
        save_attention_weights(attention_weights_data, args.experiment_name, timestamp, model_dir)
        print("Attention weights saved!")

    if args.eval_model:
        num_games = 2000
        print("Sampling games...")
        games = evals.sample_games(model, 1, num_games)
        print("Evaluating games...")
        error_rates = evals.eval_model(games)
        for eval in error_rates:
            print(f"{eval} : {error_rates[eval]}")
        #save error rates to disk
        save_evals(error_rates, args.experiment_name, timestamp, model_dir)
        
            

def create_hooked_transformer_config(
    args: argparse.Namespace,
) -> HookedTransformerConfig:
    return HookedTransformerConfig(
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


def splice_args(args: argparse.Namespace, new_args):
    print("new args", new_args)
    print("old args", args)
    args.n_layers = new_args["n_layers"]
    args.n_heads = new_args["n_heads"]
    args.d_model = new_args["d_model"]
    args.d_head = new_args["d_head"]
    args.d_mlp = new_args["d_mlp"]
    args.act_fn = new_args["act_fn"]
    args.normalization_type = new_args["normalization_type"]
    args.d_vocab = 11
    args.d_vocab_out = 10
    args.n_ctx = 10
    args.init_weights = True
    args.device = new_args["device"]
    args.seed = new_args["seed"]
    return args


def check_args(args: argparse.Namespace):
    assert args.d_model % args.d_head == 0


def make_timestamp():
    t = datetime.now()

    return f"{t.year}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"


def save_config(args: argparse.Namespace, timestamp: str, model_dir: str) -> None:
    with open(f"{model_dir}{args.experiment_name}-{timestamp}.json", "w+") as f:
        json.dump(vars(args), f, indent=4)


def load_config(model_name: str, model_dir: str):
    with open(f"{model_dir}{model_name}.json", "r") as f:
        return json.load(f)


def save_weights(
    m: HookedTransformer, experiment_name: str, timestamp: str, model_dir: str
):
    torch.save(m.state_dict(), f"{model_dir}{experiment_name}-{timestamp}.pt")


def load_weights(model_name: str, model_dir: str):
    return torch.load(f"{model_dir}{model_name}.pt")


def save_training_data(
    df: pd.DataFrame, experiment_name: str, timestamp: str, model_dir: str
):
    df.to_csv(f"{model_dir}{experiment_name} training data-{timestamp}.csv")

#save evals
def save_evals(evals: dict, experiment_name: str, timestamp: str, model_dir: str):
    with open(f"{model_dir}{experiment_name} evals-{timestamp}.json", "w+") as f:
        json.dump(evals, f, indent=4)

#save attention weights
def save_attention_weights(attention_weights: list, experiment_name: str, timestamp: str, model_dir: str):
    """Save attention weights (pos_embed, W_K, W_Q) to disk"""
    torch.save(attention_weights, f"{model_dir}{experiment_name} attention_weights-{timestamp}.pt")

# TODO: load in training data if we're fine tuning, and append the new data to old dataframe


def get_most_recent_file(directory: str):
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

    most_recent_file = files[0][:-3]  # The most recent file without extension
    return most_recent_file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_name", type=str)
    ap.add_argument("gametype", type=str, choices=["strat", "all", "minimax all", "prob all"])

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
    ap.add_argument("--save_losses", action="store_true")
    ap.add_argument("--save_checkpoints", action="store_true")
    ap.add_argument("--eval_model", action="store_true")
    ap.add_argument("--save_attention_weights", action="store_true")

    main(ap.parse_args())
