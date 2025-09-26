# Introduction

This project is based on **Walmart Recruiting - Store Sales Forecasting**, which aims to accurately predict sales across Walmart stores.

During the work, we explored and tested many well-known methods for time series forecasting, including:

* **Classical statistical models:** ARIMA, SARIMA, SARIMAX
* **Tree-based models:** XGBoost, LightGBM
* **Deep learning models for time series:** PathTST, N-BEATS, TFT, DLinear

In parallel with testing existing models, we also developed several ensembling ideas and created our own `group_stat` models, whose main principle is to combine different approaches. Several improved versions of these models were also built.

Each method and model will be discussed in detail in the following sections.

> **Note:** The detailed explanations in Georgian can be found in the `README-ka.md` file.

# EDA (Exploratory Data Analysis)

The process of Exploratory Data Analysis (EDA) is detailed in the `notebooks/01_eda.ipynb` file. The analysis revealed several important patterns and characteristics:

* **Trend and seasonality:** The data does not exhibit a pronounced long-term trend (e.g., increasing or decreasing sales year over year). However, strong **seasonality** is clearly visible — the sales charts repeat almost identically each year.

* **Department similarity across stores:** It was observed that the sales patterns of a specific department are very similar across different stores. For example, toy department sales peak in winter in all stores. The main difference is only in the sales volume, likely due to store size or regional population. Essentially, the shape of the function is identical and only shifted by a constant (`offset`).

* **Similarity of average sales across stores:** A similar pattern is seen in the average revenue of individual stores. Almost every store experiences a peak during the holiday season, while sales remain at a similar level during other periods.

* **Autocorrelation and lag features:** Analysis of the autocorrelation function showed that current week sales depend significantly on sales from the previous 2, 3, and 4 weeks (short-term dependence), as well as on sales from the same week in the previous year (long-term, seasonal dependence). This provides a basis for creating both short-term (`lag`) and long-term features for the model. For example, a meaningful feature could be the average sales for a specific week of the year (e.g., week 5) across previous years.

These observations led us to the main idea: instead of relying solely on historical averages, we can create more dynamic features. Specifically, for each data point, we add two new features:

1. **`dept_predicted_sales`**: A prediction made only at the department level (aggregating data across all stores).  
2. **`store_predicted_sales`**: A prediction made only at the store level (aggregating data across all departments).  

To obtain these features, we first train separate models to predict `store_avg` and `dept_avg` sales. Then, we use these predictions as powerful features and add them to our main model. This approach forms the foundation of our `group_stat` model.

# Models Overview

In this section, we review all the models tested in the project, ranging from classical statistical methods to modern deep learning architectures.

## Statistical Models

For the three models discussed below, we applied the following feature engineering techniques:

* **Removal of uninformative features:** The `MarkDown` columns contained little useful information, so they were removed.  
* **Encoding categorical variables:** Since linear models work only with numerical data, categorical features (`IsHoliday`, `Type`) were encoded using `one-hot encoding`.  

### ARIMA
ARIMA is one of the simplest and most classical statistical models for time series analysis. Its performance was **4195.88** on the validation set and **2660.10** on the training set.  

This is a relatively weak result, which was expected, as the model has several key limitations:  

* **Linearity:** ARIMA relies on linear assumptions, while our data clearly exhibits non-linear patterns.  
* **Ignoring exogenous features:** The model does not incorporate additional features such as holidays, temperature, or other external factors.  

### SARIMA
SARIMA is an enhanced version of ARIMA that additionally accounts for **seasonality**. Since our data exhibits annual seasonal patterns, this model is expected to perform better.  

The training runtime was very long, so we decided to experiment on a subset of the data (`sampling`) — randomly selecting a few `store/dept` pairs. The results on the sampled data were: **Train WMAE: 21910.49**, **Valid WMAE: 22466.55**.  

Although theoretically more powerful, running this model on the full dataset and making predictions is practically infeasible due to computational complexity.  

### SARIMAX
We also tested the SARIMAX model on the same sampled data. SARIMAX extends SARIMA by allowing the inclusion of exogenous features (`eXogenous features`). The results were: **Train WMAE: 578148.08**, **Valid WMAE: 308131.33**.  

As we can see, the performance significantly deteriorated compared to SARIMA. This suggests that including additional features reduced model quality. Possible reasons include:  

* SARIMAX may be too rigid to capture complex dependencies present in the data.  
* Additional features may have introduced "noise" if not used correctly.  
* The small size of the sampled dataset may have amplified overfitting when more variables were added.  

Results can be found in: `notebooks/02_linear_models.ipynb`.

## Tree-Based Models

### XGBoost

We tested XGBoost in this project, which is one of the most powerful tree-based models. It is worth noting that LightGBM follows a similar principle but is more efficient on large datasets. At this stage, we focused on XGBoost.  

The main limitation of XGBoost is that it does not naturally account for the time series structure — that is, the direct dependence of future events on past events. However, this can be mitigated with proper feature engineering.  

We added the following features:  
* Week of the year  
* Holiday indicator  
* Time remaining until the next holiday  
* Fourier transform features (to capture seasonality)  

With these features, we achieved:  
- **Train WMAE:** 1632.75  
- **Valid WMAE:** 2893.66  

This performance significantly surpasses that of the statistical models.  

### XGBoost with Autoregressive Approach

Next, we tried an autoregressive modeling approach using XGBoost. In this setup, we added `lag` features: sales from the previous 2, 3, and 51 weeks. During prediction, these `lag` features were dynamically updated using our previous predictions.  

The results were:  
- **Train WMAE:** 3001.25  
- **Valid WMAE:** 3232.62  

As we can see, the performance **worsened**. This indicates that errors made by the model in one step propagate forward as new `lag` features, which accumulate and degrade overall accuracy. This issue is particularly problematic for `store/dept` pairs with limited data, where inaccurate predictions are inevitable and relying on these errors misguides the model in subsequent forecasts.

## Neural Networks (Deep Learning Models)

Within the project, we also tested four neural network architectures specifically designed for time series forecasting. Hyperparameter tuning was performed for each model, resulting in interesting outcomes.

Below are the main hyperparameters used for each model:

| Model | Key Hyperparameters |
| :--- | :--- |
| **N-BEATS** | `input_size=52`, `h=53`, `learning_rate=1e-3`, `batch_size=256`, `optimizer=AdamW`, `shared_weights=True` |
| **D-Linear** | `input_size=60`, `h=53`, `learning_rate=1e-2`, `batch_size=512`, `optimizer=Adagrad`, `scaler_type='robust'` |
| **PatchTST** | `input_size=52`, `h=53`, `dropout=0.2`, `batch_size=64`, `activation='relu'` |
| **TFT** | `input_size=60`, `h=53`, `dropout=0.1`, `max_steps=20*104` |

*Note: `h=53` indicates that the model predicts the next 53 weeks.*

### Results (Validation WMAE)

It is worth noting that the `neuralforecast` library does not compute training errors, so only validation results are presented.

| Model | Validation WMAE |
| :--- | :--- |
| **PatchTST** | **1526.46** |
| **N-BEATS** | 1587.59 |
| **D-Linear** | 1598.11 |
| **TFT** | 1717.15 |

### Analysis

As the results show, the best performance was achieved by **PatchTST**, which significantly outperformed all previous models (both statistical and tree-based). N-BEATS and D-Linear followed closely, while TFT performed comparatively worse.  

The most interesting and somewhat surprising observation is that these results were achieved **without any external features**. The neural networks were able to capture complex seasonal patterns and dependencies using only historical time series data (univariate approach). This indicates that the dataset contains sufficient information for high-accuracy forecasting, and that analyzing only past sales dynamics may be enough to achieve strong results.

# Neural Network Ensemble

Analyzing the individual results of the neural networks revealed an interesting trend. When comparing the forecasted graphs to the actual sales (`Weekly_Sales`), we observed the following:

* The **N-BEATS** model consistently made "conservative" predictions, systematically underestimating actual sales.  
* In contrast, the **D-Linear** model exhibited the opposite behavior, systematically overestimating actual sales.  
* Other models, such as **PatchTST**, were more unstable — sometimes underestimating, sometimes overestimating.  

These observations motivated the creation of an ensemble model. The logic is simple: if one model tends to underestimate while another tends to overestimate, averaging their predictions will likely balance the errors and produce a result closer to reality.  

As a result, we created the **`nn_ensemble`** model, which is the arithmetic mean of the predictions from all four neural networks (PatchTST, N-BEATS, D-Linear, TFT).  

The validation results fully met our expectations:  

- **`nn_ensemble` Validation WMAE: 1467.43**  

This result is significantly better than that of any individual neural network. By creating the ensemble, we were able to neutralize systematic errors from individual models, yielding a more stable and accurate forecast. This once again confirms that combining models with different approaches is often the best strategy.

# GroupStat: Group Statistics-Based Model

As mentioned in the EDA section, our main hypothesis was that the `Store` and `Dept` features by themselves (as numerical identifiers) are not very informative. Much more important is to know:  
1. What were the average sales of a specific **store** during the same week?  
2. What were the average sales of a specific **department** across other stores in the same week?  

Since this information is unknown for the forecast period, we decided to predict it using separate models. Thus, the GroupStat model follows a three-step structure:  
1. Forecast average sales per store.  
2. Forecast average sales per department.  
3. Use these two forecasts as new features in the final model.  

## Feature Engineering

For all models used at each stage, we added the following features to the raw data:  
* Week of the year  
* Holiday indicator  
* Time remaining until the next holiday  
* Fourier transform features (to capture seasonality)  
* Historical average sales for the given week of the year (e.g., average sales in week 5 across previous years)  

It is important to note that the `Store` and `Dept` columns were converted from `int` to `category` type so that the model would not interpret their ordinal value (i.e., 1 < 2 conveys no meaning). Also, we did not use the `Markdown` columns, as the large number of unknown values could potentially degrade performance.

## Stage 1: Forecasting Average Sales per Store

At this stage, we aggregated the data by store to obtain each store's average weekly sales. Additionally, we created a feature **store total revenue** (from the training data) to help the model understand how "large" each store was.

The XGBoost model selected via cross-validation used the following parameters:

| Parameter | Value |
| :--- | :--- |
| `objective` | `reg:squarederror` |
| `n_estimators` | 200 |
| `learning_rate` | 0.1 |
| `max_depth` | 7 |
| `subsample` | 0.6 |
| `colsample_bytree` | 1.0 |
| `min_child_weight`| 5 |

**Results:**  
* **Validation WMAE:** 800.06 (MAE: 786.72)  
* **Train WMAE:** 253.43 (MAE: 254.05)  

*Analysis: The model learns well (low train error) and shows strong performance on validation data, indicating its effectiveness.*

## Stage 2: Forecasting Average Sales per Department

Similarly, we aggregated the data by department and added a feature **department total revenue**. In this case, only the `IsHoliday` feature from the raw data was kept, as other features (e.g., temperature, fuel price) were specific to individual stores and not to departments in general.

The XGBoost model selected via cross-validation used the following parameters:

| Parameter | Value |
| :--- | :--- |
| `objective` | `reg:squarederror` |
| `n_estimators` | 300 |
| `learning_rate` | 0.1 |
| `max_depth` | 7 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.5 |
| `min_child_weight`| 1 |

**Results:**  
* **Validation WMAE:** 1060.54 (MAE: 1017.28)  
* **Train WMAE:** 263.01 (MAE: 276.83)  

*Analysis: The performance is slightly weaker than `store_avg`, which was expected because this model had fewer informative features. Nevertheless, the results are still satisfactory. XGBoost was preferred because, unlike the neural networks discussed earlier, it allowed us to generate predictions on the training data, which was necessary for the next stage.*

## Stage 3: Global Model

In the final stage, we added the predictions from the previous two stages (`store_predicted_sales` and `dept_predicted_sales`) as powerful new features. Due to the large size of the dataset, we used **LightGBM** instead of XGBoost, which is faster.

The LightGBM model selected via cross-validation used the following parameters:

| Parameter | Value |
| :--- | :--- |
| `objective` | `regression` |
| `n_estimators` | 1000 |
| `learning_rate` | 0.1 |
| `max_depth` | 10 |

**Results:**  
* **Train WMAE:** 1156.92 (MAE: 1127.47)  
* **Validation WMAE:** 1953.42 (MAE: 1905.55)  

**Note:** With minor subsequent adjustments, the model's performance was improved to **1821 WMAE**. Both versions of the code are available in the repository:  
* `models/walmart_group_sales.py` (v1)  
* `models/walmart_group_salesv2.py` (v2)

# Final Model: Hybrid Ensemble

The analysis of the two best approaches we discussed — the neural network ensemble and the GroupStat model — highlighted their unique strengths and weaknesses:

* **Neural Network Ensemble (`nn_ensemble`)** achieves remarkable performance using only the time series data. It captures complex seasonal patterns and trends very well, but completely ignores external factors (e.g., fuel prices, unemployment rates).

* **GroupStat Model**, on the other hand, is built to leverage external features effectively. Its strength lies in accounting for store and department context, but it naturally does not capture temporal dependencies.

These observations led to a logical conclusion: to create a **hybrid model** that combines the best qualities of both approaches. The idea is to produce a final forecast that merges the predictions from these two fundamentally different models. This way, our final model simultaneously considers the internal dynamics of the time series as well as contextual external factors.

The final forecast is computed as the arithmetic mean of the predictions from the two models. We selected the weights as **0.5–0.5**, which is a simple and effective starting point. Of course, these weights can be adjusted to find an optimal balance, but due to the computational complexity of the models, this process can be time-consuming.

As a result, we obtained a model whose validation score is similar to that of the `nn_ensemble`. It is worth noting that further significant reduction in validation error (WMAE) would likely be difficult within the scope of this project without major changes (e.g., fundamentally different modeling approaches), since the achieved performance is already strong. At the same time, the hybrid model proved much more robust against **overfitting**, meaning that our final model is better generalized and is likely to perform more reliably on unseen, new data.

## How the Final Model Works (Step by Step)

To summarize, our final model operates according to the following algorithm:

1. **Step 1: Neural Network Ensemble Prediction (Result A)**
    * Four different neural networks (PatchTST, N-BEATS, D-Linear, TFT) make predictions using only historical sales data.
    * The results of these four predictions are averaged to obtain a single, stable forecast.

2. **Step 2: GroupStat Model Prediction (Result B)**
    * The average sales for stores and departments are predicted using separate models.
    * These predictions, along with other external features, are used by the global LightGBM model to generate the final forecast.

3. **Step 3: Final Hybrid Forecast**
    * The two results obtained above (A and B) are combined using a simple formula:
        `Final Prediction = (Result A * 0.5) + (Result B * 0.5)`

This hybrid approach ensures that our model is accurate, stable, and robust.

# Results

At the end of the project, we submitted predictions from our two best models to the Kaggle platform to evaluate their real-world performance on both the Public and Private leaderboards.

| Model | Public Score (WMAE) | Private Score (WMAE) |
| :--- | :--- | :--- |
| Neural Network Ensemble | 2729.61 | 2635.51 |
| **Final Hybrid Model** | **2691.28** | **2587.67** |

As the results clearly show, our **final hybrid model** outperforms the individual models on both leaderboards. This confirms our main hypothesis: incorporating the GroupStat model, which accounts for external features, significantly enhances the neural network ensemble that relies solely on time series data.






