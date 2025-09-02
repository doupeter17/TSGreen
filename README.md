# 🌱 Greenhouse Energy Cost Consumption Prediction

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
   - Read raw sensor data (greenhouse's climate data)
   - Clean missing/extreme values, create feature-engineered time-series

2. **Exploratory Data Analysis**
   - Visualize relationships (e.g. energy vs. outside temp, humidity)
   - Statistical correlation, trend detection.
   - 
3. **Exploratory Data Analysis**
   - Visualize relationships (e.g. energy vs. outside temp, humidity)
   - Statistical correlation, trend detection.

4. **Model Construction**
   - **ML model**: MLP or tree-based/ensemble model (LightGBM/XGBoost/RandomForest) as a fallback or hybrid approach
   - **DL model**: LSTM-D, GRU-D - popular model handle time-series data with missing values.

4. **Forecasting**
   - Ingest a day/week sensor behaviour/signal forecast (greenhouse's climate, sensor's activity)
   - Use models to forecast daily/weekly resource consumption.

5. **Estimation**


