# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'eddy_src/q_learning_stock'))
	print(os.getcwd())
except:
	pass

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import indicators
import argparse

def lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32, input_shape=(lookback, feature_count)))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def create_dataset(data,lookback):
    """process the data into n day look back slices
    """
    X,Y = [],[]
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback), 1:])
        Y.append(data[(i+lookback), 0])
    return np.array(X),np.array(Y)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--stock", required=True)
arg_parser.add_argument("--stock_file", required=True)
args = arg_parser.parse_args()

stock_name= args.stock
stock_file =  args.stock_file
df = pd.read_csv('data/rl/{stock_file}/{stock_name}.csv'.format(stock_file=stock_file, stock_name=stock_name), usecols=['Date','Close'], parse_dates=['Date'])
df = df[df['Close'] != 0]
df['EMA'] = indicators.exponential_moving_avg(df, window_size=10, center=False)
df['MACD_Line'] = indicators.macd_line(df, ema1_window_size=10, ema2_window_size=20, center=False)
df['MACD_Signal'] = indicators.macd_signal(df, window_size=10, ema1_window_size=10, ema2_window_size=20, center=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.index = df['Date']
df.drop('Date', axis=1, inplace=True)
feature_count = df.values.shape[1] - 1
df.head(5)

#%%
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(df.values)
print(X.shape)
# 80:20 split
train_split_len = int(len(df)/10*8)
train = X[:train_split_len, :]
test = X[train_split_len:, :]
lookback = 7

train_x, train_y = create_dataset(train, lookback)
test_x, test_y = create_dataset(test, lookback)
print(train_x.shape)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], feature_count))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], feature_count))

#%%
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001)

model = lstm_model()

history = model.fit(train_x,train_y,
                    epochs=200, verbose=0, batch_size=16,
                    shuffle=False, validation_data=(test_x, test_y))

#%%
# Save model and weights
with open("data/rl/{stock_file}/lstm/stock_pred_{stock_name}.json".format(stock_file=stock_file, stock_name=stock_name), "w") as json_file:
    json_file.write(model.to_json())
model.save('data/rl/{stock_file}/lstm/stock_pred_{stock_name}.hdf5'.format(stock_file=stock_file, stock_name=stock_name))

#%%
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend()

#%%
from tensorflow.keras.models import load_model


model = load_model('data/rl/{stock_file}/lstm/stock_pred_{stock_name}.hdf5'.format(stock_file=stock_file, stock_name=stock_name))


print(test_x.shape)
print(test_x)
print(test_y.shape)
Xt = model.predict(test_x)
print(Xt.shape)
# print(test_y)
temp_x = test_x[:,0,:]
print(temp_x.shape)
temp_data = np.concatenate((Xt, temp_x), axis=1)
print(temp_data.shape)
temp_data2 = np.concatenate((test_y.reshape(len(test_y), 1), temp_x), axis=1)
print(temp_data2.shape)
plt.plot(scaler.inverse_transform(temp_data2)[:, 0], label='Actual')
plt.plot(scaler.inverse_transform(temp_data)[:, 0], label='Predicted')
plt.legend()
