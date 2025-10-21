# 🚲 Bike Sharing Demand Prediction using AutoGluon

This project demonstrates my ability to build, optimize, and evaluate machine learning models using **[AutoGluon](https://auto.gluon.ai/)** — an AutoML framework developed by Amazon — on the **[Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)** dataset from Kaggle.

The goal of this project is to **predict the total number of bikes rented in each hour**, based on features such as weather, temperature, season, and time.  
It’s a practical problem similar to what companies like Uber, Lyft, or Bolt face when predicting customer demand in real time.

---

## 📊 Project Overview

- **Competition:** Kaggle – *Bike Sharing Demand*
- **Library used:** AutoGluon (Tabular Prediction)
- **Language:** Python (Jupyter Notebook)
- **Environment:** Amazon SageMaker / Local Jupyter
- **Dataset size:** 10,886 rows × 12 columns
- **Goal metric:** Root Mean Squared Logarithmic Error (RMSLE)

---

## 🧩 Project Workflow

### 1. Data Preparation
- Loaded `train.csv` and `test.csv` from Kaggle.
- Parsed and extracted time features from `datetime` (hour, day, month, year).
- Encoded categorical variables (season, weather, holiday, workingday).
- Split the training set into train/validation subsets.

### 2. Exploratory Data Analysis (EDA)
- Visualized rental patterns by hour, weekday, and weather condition.
- Observed **clear daily and seasonal trends** — demand peaks in morning/evening rush hours.
- Temperature and humidity showed strong correlation with bike rentals.

### 3. Model Training with AutoGluon
- Trained multiple models using **AutoGluon TabularPredictor**.
- AutoGluon automatically selected the best models (Random Forest, XGBoost, LightGBM, CatBoost, etc.) and created an **ensemble**.
- Experimented with training time, hyperparameters, and feature selection to improve the leaderboard score.

### 4. Evaluation
- Evaluated models using RMSLE and feature importance ranking.
- Compared AutoGluon leaderboard models and manually tuned iterations.
- Selected the best-performing ensemble model for final submission.

### 5. Kaggle Submission
- Generated predictions on `test.csv`.
- Submitted results to Kaggle and achieved a public leaderboard score of **<insert your score here>**.

---

## 🧮 Example of Core Code

```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('train.csv')
test_data = TabularDataset('test.csv')

label = 'count'
predictor = TabularPredictor(label=label, eval_metric='rmsle').fit(train_data)

preds = predictor.predict(test_data)
preds.head()
```
---


##
**Mohamed Sanou**
Sotware Engineering student | AI and Fullstack dev
