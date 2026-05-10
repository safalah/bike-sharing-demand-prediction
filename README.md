# Bike Sharing Demand Prediction

Predicting hourly bike rental demand for casual and registered users using machine learning, built end-to-end from EDA to deployment-ready pipeline.

---

## Background

BikeShare Co. operates an automated bike-sharing network across multiple city stations. Riders pick up a bike at one station and return it at another. The business runs on utilization and utilization depends on having bikes where people actually want them.

The problem is fleet balance. Without demand forecasting, redistribution is reactive: too many bikes pile up at quiet stations while busy ones run dry. That means wasted redistribution trips on one end and frustrated customers on the other.

This project builds a machine learning solution to predict **hourly demand per user segment**, so operations can be planned ahead instead of scrambled after the fact.

---

## The Two-Model Approach

The dataset has two target variables: `casual` (unregistered users) and `registered` (subscribers). They use the same bikes but for completely different reasons.

|               | Casual Users                    | Registered Users             |
| ------------- | ------------------------------- | ---------------------------- |
| **Pattern**   | Peaks 12:00–15:00               | Peaks 08:00 & 17:00–18:00    |
| **Behavior**  | Recreational, weather-dependent | Commuting, highly consistent |
| **Mean/hour** | 35.8                            | 153.4                        |
| **Max/hour**  | 362                             | 876                          |
| **Skewness**  | 2.48                            | 1.55                         |

Training one model for both would force it to learn two contradictory patterns at once. Separate models give cleaner results and more actionable outputs for operations.

---

## Dataset

- **Source:** Bike Sharing Dataset (hourly records)
- **Size:** 12,165 rows × 11 columns
- **Period:** 2011–2012
- **Missing values:** 0
- **Duplicates:** 0

| Feature        | Type    | Description                                |
| -------------- | ------- | ------------------------------------------ |
| `season`       | Integer | 1=Winter, 2=Spring, 3=Summer, 4=Fall       |
| `hr`           | Integer | Hour of day (0–23)                         |
| `holiday`      | Integer | National holiday flag                      |
| `temp`         | Float   | Normalized temperature                     |
| `atemp`        | Float   | Normalized "feels like" temperature        |
| `hum`          | Float   | Normalized humidity                        |
| `weathersit`   | Integer | Weather condition (1=Clear → 4=Heavy Rain) |
| `year`         | Integer | Engineered: 0=2011, 1=2012                 |
| `casual` ★     | Integer | **Target 1**                               |
| `registered` ★ | Integer | **Target 2**                               |
| `cnt`          | Integer | Total rentals — excluded (leakage)         |

---

## Key EDA Findings

**Hour of day** is by far the strongest predictor for both targets. The hourly bar charts make the two behavioral patterns obvious at a glance.

Temperature correlates positively with demand (`+0.46` for casual, `+0.33` for registered). Humidity goes the other way. Weather condition has a negative relationship with both, but casual users are more sensitive to bad weather than registered users, who keep showing up regardless.

`temp` and `atemp` have a 0.99 correlation with each other. Both were kept in the model since tree-based models handle this without issues, but it's worth flagging.

---

## Preprocessing Pipeline

```
Raw Data
   │
   ├── Extract year from dteday (2011→0, 2012→1)
   ├── Drop: dteday, cnt (leakage), visualization helpers
   ├── Define features (8) and separate targets (casual, registered)
   ├── Train-test split: 80/20, random_state=42
   └── ColumnTransformer Pipeline
           ├── StandardScaler → temp, atemp, hum
           └── Passthrough → season, hr, holiday, weathersit, year
```

---

## Modeling

Four regression models compared using 5-Fold Cross Validation:

### Casual Users

| Model             | CV RMSE    | Test RMSE  | Test MAE   | Test R²   |
| ----------------- | ---------- | ---------- | ---------- | --------- |
| **LightGBM**      | **30.978** | **30.673** | **16.825** | **0.616** |
| Random Forest     | 32.221     | 31.397     | 17.175     | 0.597     |
| XGBoost           | 32.572     | 31.909     | 17.582     | 0.584     |
| Linear Regression | 39.813     | 40.094     | 24.451     | 0.344     |

### Registered Users

| Model             | CV RMSE    | Test RMSE  | Test MAE   | Test R²   |
| ----------------- | ---------- | ---------- | ---------- | --------- |
| **LightGBM**      | **74.788** | **72.318** | **46.729** | **0.754** |
| XGBoost           | 77.259     | 74.516     | 47.770     | 0.739     |
| Random Forest     | 81.057     | 77.746     | 47.713     | 0.716     |
| Linear Regression | 125.942    | 120.538    | 86.099     | 0.317     |

LightGBM won consistently on both targets. Linear Regression fell far behind, the relationships here aren't linear and tree-based models are a better fit.

---

## Hyperparameter Tuning

GridSearchCV with 5-Fold CV across 54 parameter combinations (270 fits per model).

| Parameter       | Casual Model  | Registered Model |
| --------------- | ------------- | ---------------- |
| `n_estimators`  | 100           | 100              |
| `learning_rate` | 0.05          | 0.10             |
| `max_depth`     | -1 (no limit) | 7                |
| `num_leaves`    | 31            | 31               |

### Final Performance (Post-Tuning, Test Set)

| Target         | RMSE   | MAE    | R²    |
| -------------- | ------ | ------ | ----- |
| **Casual**     | 31.031 | 17.119 | 0.607 |
| **Registered** | 71.785 | 46.612 | 0.758 |

Registered users are easier to predict (R² 0.758 vs 0.607). That tracks commuters follow a reliable daily schedule, while casual riders depend on weather, mood, and free time.

---

## Feature Importance

Both models rank **`hr` (hour)** as the dominant feature, well above everything else.

**Casual model:** `hr` > `hum` > `temp` > `atemp` > `season` > `weathersit` > `year` > `holiday`

**Registered model:** `hr` > `hum` > `temp` > `atemp` > `season` > `weathersit` > `year` > `holiday`

Weather features (hum, temp) rank higher for casual users relative to their share — consistent with their weather-sensitive recreational behavior.

---

## Business Impact Estimate

Assumptions: redistribution cost = Rp 50,000/bike/day, model reduces unnecessary redistribution by 20%, fleet size = 1,000 bikes.

**Estimated savings: ~Rp 10,000,000/day → ~Rp 3.6 Billion/year**

Beyond cost savings, the operational shift from reactive to proactive redistribution reduces customer-facing failures during peak hours.

---

## Tech Stack

```
Python 3.9+
├── pandas, numpy              — data manipulation
├── matplotlib, seaborn        — visualization
├── scikit-learn               — pipeline, preprocessing, model selection
├── xgboost                    — XGBoost regressor
├── lightgbm                   — LightGBM regressor (final model)
└── jupyter                    — notebook environment
```

---

## Recommendations

Five things that would push this further:

1. **Weekday / weekend feature.** Casual users behave very differently on workdays vs weekends. This is probably the single highest-impact addition for improving casual model R².

2. **Station-level data.** Predicting aggregate demand is useful. Predicting per-station demand is actionable. Geographic features such as proximity to transit, CBD distance, population density would enable this.

3. **Time series approaches.** SARIMA, Prophet, or LSTM as a complementary layer to capture stronger sequential patterns in the data.

4. **Residual analysis.** The model still makes large errors at extreme hours (very early morning, peak rush). Understanding where it consistently fails determines when to override it.

5. **API deployment.** Package both models with FastAPI and integrate into the fleet management dashboard for real-time hourly predictions.

---

## Model Limitations

An R² of 0.607 for casual users means roughly 39% of variance is still unexplained. The missing pieces are likely: weekday/weekend context, station location, special events, and real-time weather data rather than daily averages.

The model was trained on 2011–2012 data. Performance on future years will degrade as city infrastructure, user habits, and fleet size change. Periodic retraining is needed.

---

_Fahrizal Safalah_
