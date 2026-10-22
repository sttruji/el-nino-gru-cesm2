# Forecasting the 2026–27 El Niño with a GRU trained on CESM2 climate simulations

This repository holds a single Jupyter notebook. It trains a **GRU** (gated recurrent unit), with a small MLP alongside it, on 23 **CESM2** climate model simulations from CMIP6. The networks predict maps of tropical-Pacific sea-surface temperature (SST) anomalies 1 to 12 months ahead, and the **Niño 3.4** index is computed from the predicted maps. The trained models are then run on observations to forecast the El Niño that is developing now. As a training-free comparison, the notebook also runs **TabPFN**, a pretrained in-context model, zero-shot on the same inputs.

Notebook: [`ECS289G_ENSO_Group5_GRU_MLP_vs_UNet.ipynb`](ECS289G_ENSO_Group5_GRU_MLP_vs_UNet.ipynb). All outputs and figures are saved in it, so it can be read on GitHub without running anything.

> **Update, 22 Oct 2026.**
> - **Bug fix:** the real-world inputs were given the wrong calendar month (the file starts in October, the code assumed January). Section 9's numbers and forecast changed; the CESM2 results did not.
> - **New Section 10:** a TabPFN comparison.

## Forecast from observations up to September 2026

These are relative Niño 3.4 values in °C: the tropical-mean (20°S–20°N) SST anomaly is removed, and the base period is 2016–2026.

| | 2026 Oct | Nov | **Dec** | 2027 Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GRU + MLP average** (our model) | +3.04 | +3.38 | **+3.55** | +3.50 | +3.20 | +2.71 | +2.09 | +1.40 | +0.63 | −0.16 | −0.89 | −1.49 |
| TabPFN (zero-shot) | +3.08 | +2.95 | **+2.60** | +2.15 | +1.59 | +1.01 | +0.59 | +0.53 | +0.20 | −0.35 | −0.76 | −1.09 |
| LIM (statistical baseline) | +2.86 | +3.10 | **+3.15** | +3.04 | +2.79 | +2.43 | +2.01 | +1.55 | +1.08 | +0.63 | +0.21 | −0.16 |

- **Our model:** a strong El Niño peaking around December 2026. Adding back the latest observed tropical-mean warming (+0.54 °C) gives a conventional Niño 3.4 of about **+4.1 °C**.
- **The other two disagree on the size of the peak.** TabPFN says the event is already near its peak, and the LIM is in between. All three turn La Niña-like by late summer 2027.
- **Our model is probably too high.** CESM2's ENSO is stronger than the real one, so our networks tend to overshoot event amplitude. In real-world hindcasts both TabPFN and the LIM have clearly lower error. Treat this as a class-project forecast, not an operational one.

## Method

| | |
|---|---|
| Training data | 23 CESM2 runs: historical r1–r11 (1850–2014) and ssp126/245/370/585 of r4, r10 and r11 |
| Inputs | SST, sea level, mixed-layer depth, zonal and meridional wind stress, precipitation: the 6 most recent months |
| Grid | 1°, 25°S–25°N × 115°E–65°W |
| Preprocessing | anomalies vs each run's 1985–2014 climatology; SST and sea level made relative (tropical mean removed); future runs detrended |
| Split | train: 1850–1955 (historical) + 2020–2100 (future scenarios); validation: r4 1961–1980; test: r4 1986–2014 |
| GRU | a small CNN encodes each month; a GRU runs over the 6 months; a linear head outputs 12 SST maps (leads 1–12) |
| MLP | the 6-month window, flattened, through one hidden layer to the same 12 maps |
| Final model | 3 seeds each of the GRU and the MLP, averaged. The validation years picked this over either model alone. |
| TabPFN (comparison) | TabPFN 3.5 with no training. The same 6-month window, compressed to each variable's leading EOFs, is the input. All 25,265 CESM2 training windows are the context. It predicts Niño 3.4 directly, not the map. |

The data setup reproduces the one our group used for its best model, a U-Net, so the two can be compared fairly. Each model is also trained with two physically motivated compound variables: the equatorial sea-level gradient and the south-east Pacific wind-stress curl. They did not improve skill.

## Results

**CESM2 test set** (r4, 1986–2014): Niño 3.4 correlation, and RMSE change relative to persistence.

| | lead 3 | lead 6 | lead 12 | lead-3 map pattern corr |
|---|---|---|---|---|
| GRU + MLP average | 0.97 / −59% | 0.92 / −64% | 0.83 / −62% | 0.82 |
| TabPFN, zero-shot | 0.97 / −63% | 0.93 / −64% | 0.83 / −61% | – (index only) |
| U-Net (group's best, reported) | 0.97 / −60% | 0.92 / −63% | 0.80 / −59% | 0.81 |
| LIM, 60 EOFs | 0.95 / −49% | 0.86 / −53% | 0.62 / −49% | 0.79 |

**Real world:** hindcasts on GODAS/ERA5 data, 2017–2026, with no fitting or tuning on this data. Niño 3.4 correlation / RMSE.

| | lead 3 | lead 6 | lead 12 |
|---|---|---|---|
| GRU + MLP average | 0.88 / 0.51 °C | 0.75 / 0.63 °C | 0.32 / 0.84 °C |
| GRU alone | 0.86 / 0.53 °C | 0.75 / 0.65 °C | 0.52 / 0.68 °C |
| TabPFN, zero-shot | 0.87 / 0.41 °C | 0.77 / 0.54 °C | 0.45 / 0.67 °C |
| LIM, 60 EOFs | 0.87 / 0.38 °C | 0.76 / 0.52 °C | 0.43 / 0.68 °C |
| Persistence | 0.71 / 0.50 °C | 0.27 / 0.79 °C | −0.16 / 1.01 °C |

- **On CESM2,** our GRU+MLP average ties the U-Net. TabPFN, with no training at all, matches both.
- **In the real world,** all the models have similar correlation. Our networks have 20–25% higher error than TabPFN and the LIM because they inherit CESM2's too-strong ENSO.
- **Small sample:** ten real-world years hold only a few ENSO events, so small differences there are not meaningful.

## Running the notebook

- **Packages:** Python with PyTorch, xarray, xESMF, intake-esm and xmip, plus `tabpfn-client` for Section 10. It was run locally on an M1 Mac with PyTorch's MPS backend. The setup cells install the packages when the notebook is opened on Google Colab.
- **CESM2 data:** downloaded automatically from the Pangeo CMIP6 catalog on Google Cloud, then cached locally (about 7 GB).
- **Course files:** two files are not included here. Place them next to the notebook:
  - `cesm2_unified.nc`: CESM2 r4 historical, 6 variables.
  - `ecs289g_realworld_inputs_CESM2.nc`: the observations for Oct 2016 – Sep 2026.
- **Training:** about 4 hours for all 12 networks on an M1 laptop with 16 GB. The weights are saved after training, so later runs only reload them.
- **TabPFN (Section 10):**
  - It runs on Prior Labs' cloud API and needs a key in the `TABPFN_TOKEN` environment variable. No key is stored in the notebook.
  - Only the compressed EOF values are uploaded. One run uses about 180k API credits.
  - The predictions are saved to `enso_tabpfn_predictions.npz`, and later runs reload them.

## Context

This is a project for the course ECS289G, group 5. Sections 1–7 of the notebook are the instructor's companion notebook. Sections 8 (CESM2), 9 (real world) and 10 (TabPFN) are our work.

Code and text were written with the help of Claude Opus 5.5 (Anthropic), as each added cell notes.
