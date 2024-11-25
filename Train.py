import tensorflow as tf
import matplotlib.pyplot as plt
from Model import build_model
from ModelDataPreprocess import download_stock_data, preprocess_data
from Metric import calculate_mape, calculate_mse, calculate_rmse

# 設定模型參數
num_layers = 1
size_layer = 128
test_size = 30
epoch = 300
dropout_rate = 0.8
learning_rate = 5e-4

if __name__ == "__main__":
    stock_list = ['2330', '2454', '2317', '2412', '2382', '2308', '2303', '2881', '2886', '2882']
    data = download_stock_data(stock_list, period="5y")
    X_train, X_test, y_train, y_test, scalers, stock_names_test = preprocess_data(data)

    # 用於測試的數據（取某個股票的數據作為範例）
    example_stock = list(data.keys())[0]
    example_df = data[example_stock]
    scaler = scalers[example_stock]

    # 建立模型
    # X_train.shape[2] 特徵數為 1 (close)
    model = build_model(num_layers, size_layer, X_train.shape[2], 1, dropout=dropout_rate)

    history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, verbose=1)

    # 保存模型
    model.save("stock_prediction_model.h5")

    # 載入模型
    loaded_model = tf.keras.models.load_model("stock_prediction_model.h5")

    # 对每支股票的测试集进行预测并计算准确率
    for stock in stock_list:
        stock_code = stock + '.TW'
        scaler = scalers[stock_code]  # 获取对应的 scaler
        stock_indices = [i for i, s in enumerate(stock_names_test) if s == stock_code]  # 获取该股票的测试集索引

        X_test_stock = X_test[stock_indices]
        y_test_stock = y_test[stock_indices]

        # 对测试集进行预测
        predictions = loaded_model.predict(X_test_stock)

        # 反向缩放预测结果和真实值
        predictions_rescaled = scaler.inverse_transform(predictions)
        y_test_rescaled = scaler.inverse_transform(y_test_stock.reshape(-1, 1))

        # 计算准确率
        mape = calculate_mape(y_test_rescaled, predictions_rescaled)
        print(f'{stock} Test mape:', mape)

        mse = calculate_mse(y_test_rescaled, predictions_rescaled)
        print(f'{stock} Test mse:', mse)

        rmse = calculate_rmse(y_test_rescaled, predictions_rescaled)
        print(f'{stock} Test rmse:', rmse)

        # 绘制测试数据和预测数据的真实数据
        test_indices = data[stock_code].index[-len(y_test_stock):]  # 测试数据的日期索引

        plt.figure(figsize=(14, 5))
        plt.plot(test_indices, y_test_rescaled, label='Test Data', color='green')  # 绘制测试数据
        plt.plot(test_indices, predictions_rescaled, label='Predicted Data', color='red')  # 绘制预测数据
        plt.title(f'{stock} Test Data and Predicted Data')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
    
    # 繪製損失圖
    plt.figure(figsize=(14, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()