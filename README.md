# 🌡️ Temperature Prediction using Machine Learning

An interactive machine learning project that predicts **temperature** using environmental factors such as humidity and pressure.

---

## 🚀 Project Overview

This project demonstrates a complete end-to-end machine learning workflow:

✅ Data Cleaning
✅ Exploratory Data Analysis (EDA)
✅ Feature Engineering
✅ Model Training
✅ Evaluation Metrics
✅ Model Saving & Deployment Ready

The notebook uses:

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib & Seaborn**

---

# 📂 Project Structure

```bash
Temperature-Prediction/
│
├── Temperature Prediction_Test (1).ipynb
├── humidity.csv
├── temperature_model.pkl
└── README.md
```

---

# 📊 Dataset Information

The dataset contains weather-related sensor readings.

### Features Used

| Feature           | Description                    |
| ----------------- | ------------------------------ |
| humidity          | Air humidity percentage        |
| pressure          | Atmospheric pressure           |
| humidity_pressure | Engineered interaction feature |
| humidity_sq       | Squared humidity feature       |
| pressure_sq       | Squared pressure feature       |

### Target Variable

| Target      | Description                 |
| ----------- | --------------------------- |
| temperature | Predicted temperature value |

---

# 🧠 Machine Learning Workflow

## 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
```

---

## 2️⃣ Load Dataset

```python
# Load dataset

df = pd.read_csv("humidity.csv")
```

---

## 3️⃣ Data Cleaning

Missing values are removed for better model accuracy.

```python
rows_before = len(df)
df_clean = df.dropna().reset_index(drop=True)
rows_after = len(df_clean)
```

---

## 4️⃣ Exploratory Data Analysis

### Humidity vs Temperature

```python
plt.figure(figsize=(8,5))
sns.scatterplot(x='humidity', y='temperature', data=df)
plt.show()
```

### Insights

* Higher humidity often affects temperature readings
* Non-linear relationships exist in weather data
* Feature engineering improves model performance

---

# ⚙️ Feature Engineering

Additional features were created to improve prediction accuracy.

```python
df_clean["humidity_pressure"] = df_clean["humidity"] * df_clean["pressure"]
df_clean["humidity_sq"] = df_clean["humidity"] ** 2
df_clean["pressure_sq"] = df_clean["pressure"] ** 2
```

---

# 🔥 Model Training

## XGBoost Regressor

```python
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Why XGBoost?

✅ Handles non-linear relationships
✅ High prediction accuracy
✅ Works well on tabular datasets
✅ Reduces overfitting

---

# 📈 Model Evaluation

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

### Metrics Used

| Metric   | Purpose                   |
| -------- | ------------------------- |
| RMSE     | Measures prediction error |
| R² Score | Measures model accuracy   |

---

# 💾 Save Model

```python
joblib.dump(model, "temperature_model.pkl")
```

The trained model can later be used for:

* Flask Deployment
* Streamlit App
* API Integration
* Real-Time Weather Prediction

---

# ▶️ How to Run the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/temperature-prediction.git
```

## 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

## 3️⃣ Run Jupyter Notebook

```bash
jupyter notebook
```

---

# 📌 Future Improvements

* Add Wind Speed Feature
* Add Rainfall Data
* Time-Series Forecasting
* Deploy Using Streamlit
* Use LSTM Deep Learning Model
* Hyperparameter Tuning

---

# 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming          |
| Pandas       | Data analysis             |
| NumPy        | Numerical operations      |
| Matplotlib   | Visualization             |
| Seaborn      | Statistical plotting      |
| Scikit-learn | ML utilities              |
| XGBoost      | Advanced regression model |
| Joblib       | Model saving              |

---

# 📸 Sample Output

```text
RMSE: 1.23
R2: 0.95
```

---

# 🤝 Contribution

Contributions are welcome.

Feel free to:

* Fork the repository
* Improve the model
* Add new features
* Create deployment support

---

# 📜 License

This project is open-source and available under the MIT License.

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub and share it with others.
