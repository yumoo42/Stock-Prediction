import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, LayerNormalization, Dropout

# 位置編碼
def position_encoding(inputs):
    T = tf.shape(inputs)[1]
    repr_dim = inputs.get_shape()[-1]
    pos = tf.reshape(tf.range(0.0, T, dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1])

# 層正規化
def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    gamma = tf.Variable(tf.ones(inputs.get_shape()[-1:]), dtype=tf.float32, name='gamma')
    beta = tf.Variable(tf.zeros(inputs.get_shape()[-1:]), dtype=tf.float32, name='beta')
    return gamma * normalized + beta

# 卷積塊
def cnn_block(x, dilation_rate, pad_sz, hidden_dim, kernel_size):
    x = LayerNormalization()(x)
    pad = tf.zeros([tf.shape(x)[0], pad_sz, hidden_dim])
    x = tf.concat([pad, x, pad], 1)
    x = Conv1D(filters=hidden_dim, kernel_size=kernel_size, dilation_rate=dilation_rate)(x)
    x = x[:, :-pad_sz, :]
    x = tf.nn.relu(x)
    return x

# 模型定義
def build_model(num_layers, size_layer, size, output_size, kernel_size=3, n_attn_heads=16, dropout=0.9):
    inputs = Input(shape=(None, size))
    x = Dense(size_layer)(inputs)
    x += position_encoding(x)
    
    for i in range(num_layers):
        dilation_rate = 2 ** i
        pad_sz = (kernel_size - 1) * dilation_rate
        x = cnn_block(x, dilation_rate, pad_sz, size_layer, kernel_size)
    
    for i in range(num_layers):
        dilation_rate = 2 ** i
        pad_sz = (kernel_size - 1) * dilation_rate
        attn_res = x = cnn_block(x, dilation_rate, pad_sz, size_layer, kernel_size)

        C = []
        for j in range(n_attn_heads):
            h_ = Dense(size_layer // n_attn_heads)(x)
            g_ = Dense(size_layer // n_attn_heads)(x)
            zu_ = Dense(size_layer // n_attn_heads)(x)
            ze_ = Dense(size_layer // n_attn_heads)(x)

            d = Dense(size_layer // n_attn_heads)(h_) + g_
            dz = tf.matmul(d, tf.transpose(zu_, [0, 2, 1]))
            a = tf.nn.softmax(dz)
            c_ = tf.matmul(a, ze_)
            C.append(c_)

        c = tf.concat(C, 2)
        x = Dense(size_layer)(attn_res + c)
        x = Dropout(dropout)(x)

    x = tf.sigmoid(x[:, -1, :])
    outputs = Dense(output_size)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss='mse')
    return model