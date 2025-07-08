from typing import Callable, Optional, Any

import einops
import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer
from copy import deepcopy
import pandas as pd


# constants
DEFAULT_LR = 1e-5
DEFAULT_WD = 1e-4
DEFAULT_EPOCHS = 40
# Super important that this is 2x2
DEFAULT_BATCH_SIZE = 4096 * 2 * 2


def rearrange(t: Tensor):
    """Formatting tensors to play nicely with F.cross_entropy.

    This can also be achieved by permuting the last two dimensions, but this should be faster.
    """
    return einops.rearrange(t, "batch seq token -> (batch seq) token")


# def train_test_split(data):


def train(
    model: HookedTransformer,
    train_data: t.Tensor,
    train_labels: t.Tensor,
    test_data: t.Tensor,
    test_labels: t.Tensor,
    optimizer: Optional[t.optim.Optimizer] = None,
    loss_fn: Callable[..., Any] = t.nn.functional.cross_entropy,
    n_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_losses: bool = True,
    save_checkpoints: bool = True,
    save_attention_weights: bool = False,
):
    """Trains models with specified data and hyperparameters.

    Test inference runs for every update on the entire set.
    """

    if save_losses:
        train_losses: list[float] = list()
        test_losses: list[float] = list()

    if save_checkpoints:
        model_checkpoints = []

    if save_attention_weights:
        attention_weights_checkpoints = []

    if optimizer is None:
        optimizer = t.optim.AdamW(
            model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WD
        )

    # TODO: should probably return losses and model once per epoch
    # TODO: something something return model_checkpoints on a log scale
    for epoch in range(n_epochs):
        for batch in range(0, len(train_data), batch_size):
            input_batch = train_data[batch : batch + batch_size]
            label_batch = train_labels[batch : batch + batch_size]

            logits_batch = model(input_batch)
            train_loss = loss_fn(rearrange(logits_batch), rearrange(label_batch))

            train_loss.backward()
            if save_losses:
                train_losses.append(train_loss.item())
            optimizer.step()
            optimizer.zero_grad()

            with t.inference_mode():
                # test inference runs for every update on the whole test set
                test_logits = model(test_data)
                test_loss = loss_fn(rearrange(test_logits), rearrange(test_labels))
                if save_losses:
                    test_losses.append(test_loss.item())
        if save_checkpoints and (epoch % 10 == 0 or epoch == n_epochs - 1):
            model_checkpoints.append({
                'epoch': epoch,
                'state_dict': deepcopy(model.state_dict()),
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item()
            })

        # Save attention weights every 10 epochs and for the last 10 epochs
        if save_attention_weights and (epoch % 10 == 0 or epoch >= n_epochs - 10):
            # Define the sequence for deterministic positional embeddings
            seq_fwd = t.tensor([[10, 0, 1, 2, 3, 4, 5, 7, 8, 6]]).to(model.cfg.device)
            
            with t.inference_mode():
                # Get resolved positional embeddings for the specific sequence
                pos_emb = model.pos_embed(seq_fwd, 0)[0]  # Shape: (10, d_model)
                
            attention_weights = {
                'epoch': epoch,
                'resolved_pos_embeddings': pos_emb.detach().clone(),
                'W_K': model.W_K.detach().clone(),
                'W_Q': model.W_Q.detach().clone(),
                'sequence': seq_fwd.cpu(),  # Save sequence on CPU for storage
            }
            attention_weights_checkpoints.append(attention_weights)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | Train Loss: {train_loss.item()} | Test Loss: {test_loss.item()}"
            )

    df = pd.DataFrame({})
    if save_losses:
        print("Creating train and test loss dataframe...")
        df["test losses"] = test_losses
        df["train losses"] = train_losses
        print("Train and test loss dataframe created!")
    if save_checkpoints:
        return (model, df, model_checkpoints)
    elif save_attention_weights:
        return (model, df, attention_weights_checkpoints)
    else:
        return (model, df)
