import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 下載股票數據
def download_stock_data(stock_list, period="5y"):
    stock_no = [stock_id + '.TW' for stock_id in stock_list]
    data = {}
    for stock in stock_no:
        ticker = yf.Ticker(stock)
        df = ticker.history(period=period)
        if not df.empty:
            data[stock] = df['Close']
    return data

# 預處理單支股票數據
def preprocess_single_stock(df, time_steps=30):
    df_reshaped = df.values.reshape(-1, 1).astype('float32') # 轉為 numpy array 並改變形狀

    minmax = MinMaxScaler().fit(df_reshaped)
    scaled_data = minmax.transform(df_reshaped)  # 縮放到 [0, 1] 範圍
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # X.shape (1187, 30, 1)
    # y.shape (1187,)
    return X, y, minmax

# 預處理多支股票數據
def preprocess_data(data, time_steps=30):
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    scalers = {}
    stock_names_test = []
    
    for stock, df in data.items():
        X, y, scaler = preprocess_single_stock(df, time_steps)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        
        scalers[stock] = scaler # 保存每支股票的縮放器
        stock_names_test.extend([stock] * len(y_test)) # 保存測試集對應的股票名稱
    
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    # X_train.shape (9490, 30, 1)
    # X_test.shape (2380, 30, 1)
    # y_train.shape (9490,)
    # y_test.shape (2380,)
    return X_train, X_test, y_train, y_test, scalers, stock_names_test