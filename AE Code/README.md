## This Part of the Repo Documents the Use of an Autoencoder (AE) to supplement the Top-K operation. Note that this is an extension of the base project and is not included in the main report.

### Method 
Here, we introduce an addition term into the standard top-K loss function. 

The Standard top-k Loss function is: $L = \alpha*L_{CE} + (1- \alpha)*D_{KL}(P_{Teacher-Top-k}||P_{Student-Top-k})$ 

Here, we add a term to the standard top-k Loss function to get: $L = \alpha *L_{CE} + (1- \alpha)*L_{hybrid}$ 

Where $L_{hybrid}$ = $0.7 * D_{KL}(P_{Teacher-Top-k}||P_{Student-Top-k})$ + $0.3 * D_{KL}(P_{Teacher-Compressed}||P_{Student-Compressed})$

Here, $P_{Teacher-Compressed} = Encoder(Full-Teacher-Logits)$ and $P_{Student-Compressed} = Encoder(Full-Student-Logits)$. That is, the full teacher and student logits are compressed into an AE latent space of dimenion = L. Here, we consider L=8 for the latent dimension. 

Note: Our AE was trained in this notebook: https://colab.research.google.com/drive/10m2bb0JRg7G8Xf0PT2aL0DMEMp5FPlti?usp=sharing 

### Experiment 
To test our AE-based KD pipeline, we run top-8 student training using the new loss function give above. Here, our training budget cost will be 3K = 3*8 = 24 because training a single token will require storing the top-8 logits, top-8 token indices, and the L=8 compressed logits.

### Results and Analysis

| Method | Budget (per token) | Best PPL |
|--------|-------------------|----------|
| **Top-K (K=16)** | *32* | 41.91 |
| **Top-K (K=8)** | 16 | **45.23** |
| **Hybrid-KD (K=8)** | *24* | **42.37** |

Note from the above table that our method (Hybrid-KD with K=8) does outperform vanilla top-K (K=8). However, using Hybrid-KD requires a larger teacher budget (24 versus 16), so it is actually more helpful to compare our Hybrid-KD with Top-K (K=16). Note that, within the bounds of statistical significance, Hybrid-KD is able to reach the same perplexity level as Top-K (K=16). **This result indicates that a hybrid loss combining both top-K and compression terms can achieve a given level of perplexity (here, ~42) with a smaller budget (24 instead of 32). This can be done by factoring a top-K (large K, here 16) into a top-(K/2) with L-dimensional compression (here, K/2=8, L=8)**.

