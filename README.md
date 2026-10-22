# Forecasting the 2026–27 El Niño with a GRU trained on CESM2 climate simulations

This repository holds a single Jupyter notebook. It trains a **GRU** (gated recurrent unit), with a small MLP alongside it, on 23 **CESM2** climate model simulations from CMIP6. The networks predict maps of tropical-Pacific sea-surface temperature (SST) anomalies 1 to 12 months ahead, and the **Niño 3.4** index is computed from the predicted maps. The trained models are then run on observations to forecast the El Niño that is developing now.

Notebook: [`ECS289G_ENSO_Group5_GRU_MLP_vs_UNet.ipynb`](ECS289G_ENSO_Group5_GRU_MLP_vs_UNet.ipynb). All outputs and figures are saved in it, so it can be read on GitHub without running anything.

## Forecast from observations up to September 2026

These are relative Niño 3.4 values: the tropical-mean (20°S–20°N) SST anomaly is removed, and the base period is 2016–2026.

| 2026 Oct | Nov | **Dec** | 2027 Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| +2.96 | +3.22 | **+3.30** | +3.19 | +2.92 | +2.49 | +1.91 | +1.25 | +0.54 | −0.13 | −0.72 | −1.22 |

- **Peak:** a strong El Niño peaking around December 2026.
- **Conventional index:** adding back the latest observed tropical-mean warming (+0.54 °C) gives a conventional Niño 3.4 of about **+3.8 °C**.
- **After the peak:** the event decays through spring 2027 and turns La Niña-like by late summer 2027.
- **Likely too high:** CESM2's ENSO is stronger than the real one, so these models tend to overshoot event amplitude. In real-world hindcasts the statistical LIM baseline has lower error. Treat this as a class-project forecast, not an operational one.

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

The data setup reproduces the one our group used for its best model, a U-Net, so the two can be compared fairly. Each model is also trained with two physically motivated compound variables: the equatorial sea-level gradient and the south-east Pacific wind-stress curl. They did not improve skill.

## Results

**CESM2 test set** (r4, 1986–2014): Niño 3.4 correlation, and RMSE change relative to persistence.

| | lead 3 | lead 6 | lead 12 | lead-3 map pattern corr |
|---|---|---|---|---|
| GRU + MLP average | 0.97 / −59% | 0.92 / −64% | 0.83 / −62% | 0.82 |
| U-Net (group's best, reported) | 0.97 / −60% | 0.92 / −63% | 0.80 / −59% | 0.81 |
| LIM, 60 EOFs | 0.95 / −49% | 0.86 / −53% | 0.62 / −49% | 0.79 |

**Real world:** hindcasts on GODAS/ERA5 data, 2017–2026, with no fitting or tuning on this data. Niño 3.4 correlation / RMSE.

| | lead 3 | lead 6 | lead 12 |
|---|---|---|---|
| GRU + MLP average | 0.88 / 0.47 °C | 0.75 / 0.59 °C | 0.33 / 0.79 °C |
| GRU alone | | | 0.52 / 0.64 °C |
| LIM, 60 EOFs | 0.87 / 0.38 °C | 0.76 / 0.52 °C | 0.43 / 0.68 °C |
| Persistence | 0.71 / 0.50 °C | 0.27 / 0.79 °C | −0.16 / 1.01 °C |

## Running the notebook

- **Packages:** Python with PyTorch, xarray, xESMF, intake-esm and xmip. It was run locally on an M1 Mac with PyTorch's MPS backend. The setup cells install the packages when the notebook is opened on Google Colab.
- **CESM2 data:** downloaded automatically from the Pangeo CMIP6 catalog on Google Cloud, then cached locally (about 7 GB).
- **Course files:** two files are not included here. Place them next to the notebook:
  - `cesm2_unified.nc`: CESM2 r4 historical, 6 variables.
  - `ecs289g_realworld_inputs_CESM2.nc`: the observations for Oct 2016 – Sep 2026.
- **Training:** about 4 hours for all 12 networks on an M1 laptop with 16 GB. The weights are saved after training, so later runs only reload them.

## Context

This is a project for the course ECS289G, group 5. Sections 1–7 of the notebook are the instructor's companion notebook. Sections 8 (CESM2) and 9 (real world) are our work.

Code and text were written with the help of Claude Opus 5.5 (Anthropic), as each added cell notes.
