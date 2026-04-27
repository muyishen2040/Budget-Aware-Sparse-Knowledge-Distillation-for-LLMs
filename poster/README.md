# Poster Plotting Code

## Overview

This folder contains the visuals from our poster, as well as the code we used to generate the validation PPL and NLL plots in our poster.

## Outputs

- `poster_plots/` contains the plots featured in our poster.
- `all_methods_plots/` contains the same plots with the Top-8 + Autoencoder Hybrid extension included.

## All-Methods Plots

![Validation PPL vs budget](all_methods_plots/validation_ppl_vs_budget.png)

![Validation NLL vs budget](all_methods_plots/validation_nll_vs_budget.png)

## Full Project Poster
[Poster PDF](https://drive.google.com/file/d/18p-Zd08L1cyRhnDK7ykYS1KKHRs7v0JL/view?usp=sharing)


## Regenerating Plots

From the repository root:

```bash
python poster/plotter.py
```

## Diagrams

The illustrations of our adaptive sparse methods were manually drawn in Notability.

### Figure 1: Adaptive Top-K

![Adaptive Top-K](poster_diagrams/adaptive_top_k.png)

### Figure 2: Adaptive Top-K + Head Mass Weighting

![Adaptive Top-K + Head Mass Weighting](poster_diagrams/head_mass_weighting.png)
