#  Seafood Price Prediction (Time Series Forecasting)

##  Project Overview
This project aims to **analyze and forecast seafood prices over time** using **time series modeling techniques**.  
The dataset used contains historical seafood prices in **USD**, which are converted into **INR** to make the analysis regionally relevant.  
The notebook walks through data exploration, visualization, and prediction using multiple forecasting approaches to estimate **future seafood demand and price trends**.

---

##  Files Included
| File | Description |
|------|--------------|
| `seafood price prediction.ipynb` | Jupyter Notebook containing data preprocessing, visualization, model training, and forecasting. |
| `seafood_prices.csv` | The dataset used for training and testing (upload this file in your repository). |

---

##  Key Steps in the Notebook

### 1. **Data Loading & Conversion**
- Imports the seafood dataset from CSV.
- Converts prices from **USD to INR** using a fixed conversion rate.
- Parses dates and organizes the data for time series analysis.

### 2. **Exploratory Data Analysis (EDA)**
- Visualizes seafood price fluctuations over time.
- Identifies **trends**, **seasonality**, and **outliers**.
- Calculates rolling averages to understand long-term changes.

### 3. **Data Preprocessing**
- Handles missing values and inconsistent data points.
- Sets the `Date` column as the time index.
- Prepares the data for model training and forecasting.

### 4. **Model Building**
Multiple forecasting models are applied and compared to identify the best performer:
- **ARIMA Model** — captures linear trends and seasonality in prices.
- **SARIMA Model** — enhances ARIMA by adding seasonal components.
- **Prophet Model (Meta/Facebook)** — handles seasonality, holidays, and long-term trends effectively.
- **LSTM (Neural Network)** — captures complex nonlinear relationships in price fluctuations.

### 5. **Model Evaluation**
- Evaluates models using **MAE**, **RMSE**, and **R² Score**.
- Compares prediction accuracy across models.
- Selects the model with the lowest forecast error as the best predictor.

### 6. **Forecasting Future Prices**
- Forecasts seafood prices for upcoming months/years.
- Visualizes forecasted trends alongside historical data.
- Provides insights into expected future demand and market prices.

---

## Results & Insights
- The dataset showed a **gradual increase in seafood prices**, influenced by seasonal demand patterns.  
- The **Prophet** and **LSTM** models provided the most accurate forecasts.  
- Forecast results show **a steady upward trend**, indicating potential price increases in the near future.

---

##  How to Run the Notebook

###  Google Colab (Recommended)
1. Open the notebook in **Google Colab**.  
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update dataset path if needed:
   ```python
   data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/seafood_prices.csv')
   ```
4. Run all cells in sequence.

###  Local Jupyter Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/reallakshay/seafood-price-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `seafood price prediction.ipynb` and execute all cells.

---

##  Technologies Used
- **Python 3.x**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **Statsmodels (ARIMA/SARIMA)**
- **Prophet**
- **TensorFlow / Keras (LSTM)**
- **scikit-learn**

---

## What the Model Does (Brief Explanation)
The model takes historical seafood price data and learns **patterns over time**, such as:
- **Trends** — gradual increase or decrease in prices.  
- **Seasonality** — recurring price changes based on time of year (e.g., festive or fishing seasons).  
- **Random Fluctuations** — short-term variations due to supply/demand shocks.

Once trained, the model uses these learned patterns to **predict future prices** — helping identify when seafood prices are likely to rise or stabilize.  
This insight can be valuable for **retailers, suppliers, and consumers** in planning purchases and pricing strategies.

---

##  Author
**Lakshay Mittal**  
 
