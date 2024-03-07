from alphatoe import models, plot, interpretability, game
import pandas as pd
import torch
from pytorch_memlab import LineProfiler, MemReporter
import einops
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import numpy as np
import tqdm

#seed
torch.manual_seed(1337)

hidden_sizes = [512]
lamdas = [2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8]
#lamdas = [2.5e-8]
results = []
batch_sizes = [int(2**9), int(2**12)]

for batch_size in tqdm.tqdm(batch_sizes):
    for hidden_size in tqdm.tqdm(hidden_sizes):
        for lamda in tqdm.tqdm(lamdas):
            autoenc = models.SparseAutoEncoder(512, hidden_size).to("cuda")
            loss_fn = torch.nn.functional.mse_loss
            optimizer = torch.optim.Adam(autoenc.parameters(), lr=1e-4)

            act_data = torch.load("data/all_games_act_data.pt")

            epochs = 601
            losses = []
            for epoch in tqdm.tqdm(range(epochs)):
                for batch in range(0, act_data.shape[0], batch_size):
                    dat = act_data[batch : batch + batch_size].to("cuda")

                    l0, reg, guess = autoenc(dat)
                    mse_loss = loss_fn(guess, dat)

                    sparse_loss = lamda * reg
                    # sparse_loss = 0
                    loss = mse_loss + sparse_loss
                    # losses.append(interpretability.numpy(loss))
                    losses.append([mse_loss.item(), sparse_loss.item(), l0.item()])
                    optimizer.zero_grad()
                    loss.backward()
                    #print(losses[-1])
                    optimizer.step()

                    with torch.no_grad():
                        last_loss = loss_fn(guess, dat, reduction="none")
                
                if epoch % 30 == 0:
                    results.append({
                        "hidden_size": hidden_size,
                        "lamda": lamda,
                        "batch_size": batch_size,
                        "epoch": epoch+1,
                        "MSE": mse_loss.item(),
                        "L1": sparse_loss.item(),
                        "L0": l0.item(),
                        'loss': loss.item()
                    })
            torch.save(autoenc.state_dict(), f"scripts/models/SAE_hidden_size-{hidden_size}_lamda-{lamda}_epoch-{epochs-1}_batch_{batch_size}.pt")
            torch.cuda.empty_cache()
df = pd.DataFrame(results)
df.to_csv("data/SAE_hyperparam_results_decoder_unitnorm_batch_size.csv")