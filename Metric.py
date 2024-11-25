import numpy as np

def calculate_mape(real, predict):
    real, predict = np.array(real), np.array(predict)
    return np.mean(np.abs((real - predict) / real)) * 100

def calculate_mse(real, predict):
    real, predict = np.array(real), np.array(predict)
    return np.mean((real - predict) ** 2)

def calculate_rmse(real, predict):
    real, predict = np.array(real), np.array(predict)
    return np.sqrt(np.mean((real - predict) ** 2))
