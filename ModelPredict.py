from ModelDataPreprocess import download_stock_data, preprocess_single_stock
import tensorflow as tf   

def Predict(stock_list):
    data = download_stock_data(stock_list, period="5y")

    for stock in stock_list:
        stock_code = stock + '.TW'
        stock_df = data[stock_code]  # 取得該股票的數據
        X, y, scaler = preprocess_single_stock(stock_df)  # 預處理數據

        # 加载模型
        loaded_model = tf.keras.models.load_model("stock_prediction_model.h5")

        # 進行預測
        predictions = loaded_model.predict(X)

        # 反向縮放預測結果和真實值
        predictions_rescaled = scaler.inverse_transform(predictions)
        return predictions_rescaled