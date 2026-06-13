<div align="center">

# ⚽ FIFA 2026 World Cup — AI Match Predictor



[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Kushagramishra458/FIFA-WORLD-CUP-MATCHES-PREDICTION/main?labpath=FINAL PREDICTION.ipynb)

## Project Overview
This project utilizes an XGBoost machine learning model to predict the outcomes of the 2026 FIFA World Cup...
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-174A7C?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-GitHub%20Open%20Data-blue?style=for-the-badge)](https://github.com/martj42/international_results)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Predicting the **outcomes of international football matches** and forecasting the 2026 World Cup group stages using XGBoost, dynamic Elo ratings, and chronological, zero-leakage feature engineering.

</div>

---

## ⚠️ Sports Analytics Disclaimer

> **This project is for educational and portfolio purposes only.** It demonstrates applied machine learning and feature engineering techniques and is not intended to be used as financial advice or for sports betting.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [The Challenge of Football Prediction](#-the-challenge-of-football-prediction)
- [Dataset](#-dataset)
- [Class Distribution](#-class-distribution)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔬 About the Project

Predicting football matches is notoriously difficult due to the low-scoring nature of the sport and the high variance of individual games. This project applies **ensemble machine learning algorithms** to automatically forecast match outcomes (Home Win, Draw, Away Win) and calculate Expected Goals (xG).

A key challenge in sports ML is **data leakage**—using future knowledge (like a team's end-of-year win rate) to predict a match in the middle of the year. This project tackles this by building a strict chronological pipeline that only calculates running form and head-to-head metrics based on the exact state of the world *prior* to kickoff.

**What this project covers:**
- Strict zero-leakage feature engineering on time-series sports data
- Custom implementation of a dynamic Elo Rating system
- Stratified cross-validation for highly imbalanced class distributions (Draws)
- SHAP-based game theory for model explainability
- Interactive UI dashboarding using IPyWidgets

---

## 🥅 The Challenge of Football Prediction

Unlike basketball or tennis, football matches frequently end in a **Draw**, creating a complex multi-class classification problem. Furthermore, predicting a simple "Win/Loss" is often insufficient. This pipeline goes a step further by utilizing a parametric model to project **Expected Goals (xG)**, balancing historical attack and defense intensities scaled by Elo differentials to provide a mathematically sound projected scoreline.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [International Football Results (1872–Present)](https://github.com/martj42/international_results) |
| **Samples** | Filtered to Modern Era (2000–Present) |
| **Features** | Elo Ratings, Rolling Form (GF/GA), H2H Win Rates, Tournament Weights |
| **Classes** | 3 (Away Win, Draw, Home Win) |
| **Challenge** | Preventing temporal data leakage while building rolling historical averages. |

---

## 📋 Class Distribution

| Code | Outcome | Description |
|:----:|-------|---------|
| 0 | **Away Win** | Visiting team wins (lowest frequency) |
| 1 | **Draw** | Match ends in a tie |
| 2 | **Home Win** | Hosting team wins (highest frequency due to home advantage) |

> **Note:** The "Home Win" class naturally dominates international football datasets due to home-field advantage. Models must be carefully validated using Macro F1-scores to ensure they don't simply predict the majority class.

---

## ⚙️ Methodology

The project follows a chronologically strict ML pipeline:

```text
Raw Match Data (1872–Present)
        │
        ▼
  Data Preprocessing & Filtering (2000+)
  ├── Handle neutral venues
  └── Clean missing scores
        │
        ▼
  Zero-Leakage Feature Engineering
  ├── Dynamic Elo Rating updates (scaled by margin of victory)
  ├── Rolling 5-match form (goals scored/conceded, points)
  └── Head-to-Head historical win rates
        │
        ▼
  Model Training & Cross-Validation
  ├── Logistic Regression
  ├── Random Forest Classifier
  └── XGBoost Classifier  ← Best Model
        │
        ▼
  Simulation & Explainability
  ├── SHAP Value extraction for global/local explainability
  └── Interactive IPyWidgets UI Deployment
