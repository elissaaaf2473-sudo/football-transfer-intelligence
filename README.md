
# Football Transfer Intelligence

An explainable machine learning project for
predicting recorded football transfer fees.

## Live Dashboard

https://elissaaaf2473-sudo.github.io/football-transfer-intelligence/

## Dataset

Soccer Player Transfer Dataset (Kaggle)

Source:
https://www.kaggle.com/datasets/otaviofbrito/soccer-player-transfer-dataset

Research sample:
29,272 recorded positive-fee, non-loan
transactions from 2010 to 2023.

## Methodology

- Temporal train-validation-test splitting
- Ridge Regression
- Random Forest Regressor
- CatBoost Regressor
- Expanding-window validation
- Hyperparameter tuning
- Ablation study
- SHAP explainability
- Player-cluster bootstrap analysis

## Final Model

Algorithm: CatBoost Regressor

- Depth: 8
- Iterations: 1000
- Learning rate: 0.04
- L2 regularization: 8
- Target transformation: log1p
- Training: 2010-2021
- Testing: 2022-2023

## Final Holdout Test

| Metric | CatBoost |
|---|---:|
| MAE | 2,307,871.22 |
| RMSE | 6,944,274.56 |
| R-squared | 0.3625 |
| MAE improvement vs median baseline | 28.72% |

## Key Findings

Club identities provide substantial
predictive information.

CatBoost outperforms Ridge Regression,
Random Forest, and the median baseline
on the final holdout test.

However, the model tends to underestimate
extremely high transfer fees.

## Limitations

- Only recorded positive-fee non-loan
  transactions are modeled.
- Zero-fee and undisclosed-fee transfers
  cannot automatically be treated equally.
- Historical features are limited.
- The model is not a causal model.
- Extreme-value prediction remains challenging.
- Early exploratory analysis included the
  holdout years, although final model tuning
  did not use holdout outcomes.

## Deployment

Interactive research dashboard deployed
using GitHub Pages.

The website visualizes saved model results.
It does not execute live Python predictions.
