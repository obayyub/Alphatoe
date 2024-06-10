from alphatoe import models, plot, interpretability, game
import pandas as pd
import torch
from pytorch_memlab import LineProfiler, MemReporter
import einops
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import numpy as np
import tqdm

torch.cuda.empty_cache()

#seed
torch.manual_seed(1337)

hidden_sizes = [1024, 512]
lamdas = [7.5e-8, 2.5e-8, 7.5e-9]
cos_lams = [0, 1e-3]
#lamdas = [2.5e-8]
results = []
batch_sizes = [int(2**12)]

for batch_size in tqdm.tqdm(batch_sizes):
    for hidden_size in tqdm.tqdm(hidden_sizes):
        for lamda in tqdm.tqdm(lamdas):
            for cos_lam in tqdm.tqdm(cos_lams):
                autoenc = models.SparseAutoEncoder(512, hidden_size).to("cuda")
                loss_fn = torch.nn.functional.mse_loss
                optimizer = torch.optim.Adam(autoenc.parameters(), lr=1e-4)

                act_data = torch.load("data/all_games_act_data.pt")

                epochs = 601
                losses = []
                for epoch in tqdm.tqdm(range(epochs)):
                    for batch in range(0, act_data.shape[0], batch_size):
                        dat = act_data[batch : batch + batch_size].to("cuda")

                        l0, reg, cos, guess = autoenc(dat)
                        mse_loss = loss_fn(guess, dat)

                        sparse_loss = lamda * reg
                        cos_loss = cos_lam * cos
                        # sparse_loss = 0
                        loss = mse_loss + sparse_loss + cos_loss
                        # losses.append(interpretability.numpy(loss))
                        losses.append([mse_loss.item(), sparse_loss.item(), l0.item(), cos_loss.item()])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            last_loss = loss_fn(guess, dat, reduction="none")
                    print(losses[-1])
                    print(cos.item())
                    
                    if epoch % 30 == 0:
                        results.append({
                            "hidden_size": hidden_size,
                            "lamda": lamda,
                            "cos_lam": cos_lam,
                            "batch_size": batch_size,
                            "epoch": epoch+1,
                            "MSE": mse_loss.item(),
                            "L1": sparse_loss.item(),
                            "L0": l0.item(),
                            'loss': loss.item(),
                            "cosine": cos.item(),
                            "cos loss": cos_loss.item(),
                        })
                torch.save(autoenc.state_dict(), f"scripts/models/SAE_hidden_size-{hidden_size}_lamda-{lamda}_batch_{batch_size}_wo_l1_cosine-{cos_lam}.pt")
                torch.cuda.empty_cache()
df = pd.DataFrame(results)
df.to_csv("data/test_cos_lam_l1_cos_small_batch.csv")