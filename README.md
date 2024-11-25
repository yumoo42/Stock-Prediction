# Stock-Prediction
## 1. Stock Data Retrieval
- Use the following URL to fetch the current list of all Taiwan Stock Exchange (TWSE) stocks: https://isin.twse.com.tw/isin/C_public.jsp?strMode=2
- Extract stock symbols and append .TW to each symbol.
- Retrieve historical stock data using the yfinance library.

## 2. AI Prediction
### (1) Data Processing
- Data Selection:
  - Select representative stocks as training data.
  - Use closing prices as the feature.
  - Use 5 years of historical data.
- Predict the 31st day’s closing price based on the previous 30 days’ closing prices.
- Dataset Splitting:
  - Split the data into training and test sets in an 8:2 ratio.
- Processing Multiple Stocks:
  - Process each stock individually by scaling the data to the range [0, 1].
  - Split the processed data into training and test sets.
  - Merge the processed data for further use.
### (2) Model Architecture
- Refer to the repository: https://github.com/huseinzol05/Stock-Prediction-Models
- Several models are available, but for this project, we selected the Dilated-CNN-Seq2Seq model due to its fast training time and high accuracy.

### (3) Training Configuration
- Epochs: 300
- Optimizer: Adam
- Learning Rate: 5e-4
- Loss Function: MSE (Mean Squared Error)
- Dropout Rate: 0.8
## 3. Prediction and Analysis
### Indicators for Analysis
- Buy Signals:
  - RSI < 30
  - Short-term RSI (7) > Long-term RSI (14)
  - MACD > MACD Signal
  - Model-predicted closing price > Current closing price
  - External volume > Internal volume, and an increase in buy orders.
- Sell Signals:
  - RSI > 70
  - Short-term RSI (7) < Long-term RSI (14)
  - MACD < MACD Signal
  - Internal volume > External volume, and an increase in sell orders.
### Model Prediction Process
- Since predicting all stocks in the TWSE would take too much time, the stocks are first filtered using the RSI-based indicators. After filtering, the model predicts the closing prices, reducing the prediction time to about 2 minutes.
- The external and internal volumes are checked after the program runs to confirm the current conditions.

### RSI and MACD Calculations
- The talib library is used for calculating RSI and MACD indicators.
- Installation of talib:
-   Visit https://github.com/cgohlke/talib-build/releases to download the appropriate version of the wheel file.
-   Use the terminal command to install the library:
```
  pip install <wheel_file_name>
```
