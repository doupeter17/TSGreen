# TSGreen - AutoML Framework for Time-series in Smart Greenhouse

**Objective**  
Use data science and optimization techniques to:
1. **Forecast daily and weekly sensor's behaviour** in a greenhouse, based on sensor and weather data to **calculate energy consumption**.
2. **Optimize resource control strategy** to minimize total energy cost (heat, co2,...) while meeting production constraints.

---

##  Background & References
Dataset is provided by Autonomous Greenhouse Challenge(AGC) 2019~2020 - 2nd Edition.

---

## Project Flow

1. **Data Ingestion & Cleaning**
   - Read raw sensor data (greenhouse's sensors data) (already have missing data)
   - Finding longest unmissing data range.
   - Create artficial missing block.
   - Auto handling missing data through collection of imputation methods.
   - Evaluate methods using MAE, MSE or R2.
   - Apply back to origin dataset.
     
2. **Exploratory Data Analysis**
   - Visualize relationships (e.g. energy vs. outside temp, humidity)
   - Statistical correlation, trend, stationary detection using statistic test (KPSS, ADF)
   - Finding relationship from multiple time-series using Granger Causality
   
3. **Feature Engineering**
   - Feature Creation: Basic statistical features, datetime features, Differencing, Shifting.
   - Feature Selection: Using feature important score from ML models.

4. **Model Construction**
   - ML Models: RandomForest, XGBoost, CatBoost
   - Auto hypertuning

5. **Estimation**
   - Popular metrics: R2, MSE, RMSE.

## Current Progress:
- In current progress, the experiment is already finished, but the project is not finished yet, I need to leverage my OOP knowledge to transfer from Jupiter notebook code to a framework that easy to use and scale. I have finished the OOP for Step 1, and continue, to observe the code, please check in directory **'Code/TSGreen.ipynb'**

Author: 
Duong Quang Thanh


