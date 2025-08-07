# Bias Detection in Neural Networks
This repository contains code to reproduce the figures from the seminar paper on fairness in neural networks.
- Written by **Clara Kümpel**  
- For the seminar **"Fairness in Algorithms"**

This project explores bias detection in neural networks using the Adult Income dataset. It includes data preprocessing, model training, and various fairness and representation analyses.

> Parts of the preprocessing and baseline analysis are adapted from the CDEI Fairness Finance tutorial:  
> https://cdeiuk.github.io/bias-mitigation/finance/

## Overview

The code is organized into notebooks for:

- Data loading and cleaning (`data_cleaning.ipynb`)
- Exploratory data analysis (`data_analysis.ipynb`)
- Model training and evaluation (`model.ipynb`)

The folder structure includes:
- `Adultdata/` – raw input data
- `artifacts/` – trained models and saved results
- `figures/` – some generated plots and visualizations
- `helpers/` – utility functions for plotting and evaluation



## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```