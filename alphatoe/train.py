from typing import Callable, Optional

import einops
import torch as t
from transformer_lens import HookedTransformer


# constants
DEFAULT_LR = 1e-5
DEFAULT_WD = 1e-4
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 4096 * 4


def rearrange(t):
    """Formatting tensors to play nicely with F.cross_entropy.

    This can also be achieved by permuting the last two dimensions, but this should be faster.
    """
    return einops.rearrange(t, "batch seq token -> (batch seq) token")


def train(
    model: HookedTransformer,
    train_data: t.Tensor,
    train_labels: t.Tensor,
    test_data: t.Tensor,
    test_labels: t.Tensor,
    optimizer: Optional[t.optim.Optimizer] = None,
    loss_fn: Callable = t.nn.functional.cross_entropy,
    n_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> HookedTransformer:
    """Trains models with specified data and hyperparameters.

    Test inference runs for every update on the entire set.
    """
    train_losses = list()
    test_losses = list()

    if optimizer is None:
        optimizer = t.optim.AdamW(
            model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WD
        )

    for epoch in range(n_epochs):
        for batch in range(0, len(train_data), batch_size):
            input_batch = train_data[batch : batch + batch_size]
            label_batch = train_labels[batch : batch + batch_size]

            logits_batch = model(input_batch)
            train_loss = loss_fn(rearrange(logits_batch), rearrange(label_batch))

            train_loss.backward()
            train_losses.append(train_loss.item())
            optimizer.step()
            optimizer.zero_grad()

            with t.inference_mode():
                # test inference runs for every update on the whole test set
                test_logits = model(test_data)
                test_loss = loss_fn(rearrange(test_logits), rearrange(test_labels))
                test_losses.append(test_loss.item())

        print(
            f"Epoch {epoch} | Train Loss: {train_loss.item()} | Test Loss: {test_loss.item()}"
        )

    return model