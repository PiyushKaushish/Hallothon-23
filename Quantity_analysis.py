import os
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

file_list = []
for dirname, _, filenames in os.walk('dataset'):
    for filename in filenames:
        file_list.append(os.path.join(dirname, filename))
# select stocks
dfs = []
for file in file_list:
    df = pd.read_csv(file)
    #print(f"File: {file}, Columns: {df.columns}, DataTypes: {df.dtypes}")
    if 'Symbol' in df.columns:

        if (df.shape[0] == 5306) & (df['Symbol'].nunique() == 1):
            dfs.append(df)

df = pd.concat(dfs, axis=0).sort_values(['Date','Symbol']).drop(['Trades','Deliverable Volume','%Deliverble'], axis=1)
#We are preforming feature engineering so that raw data can be convereted into features that can be later used for the better sentiment analysis


#creating features in a DataFrame, sorting it, and then filtering based on date. Additionally, you're checking the length of each stock based on the date range '2000-01-04' to '2021-04-29'
# Define a function named create_feature that takes a DataFrame df as input


def create_feature(df):
    # Create a new column 'Open_t+1' that contains the next day's opening prices
    df['Open_t+1'] = df['Open'].shift(-1)
    
    # Create a new column 'Close_t+1' that contains the next day's closing prices
    df['Close_t+1'] = df['Close'].shift(-1)
    
    # Calculate a factor 'factor_overnight_OHLCV_t0' based on the ratio of next day's open to current day's close
    df['factor_overnight_OHLCV_t0'] = df['Open_t+1'] / df['Close'] - 1
    
    # Calculate a factor 'factor_HL_OHLCV' based on the ratio of high to low prices
    df['factor_HL_OHLCV'] = df['High'] / df['Low'] - 1
    
    # Calculate a factor 'factor_HO_OHLCV' based on the ratio of high to open prices
    df['factor_HO_OHLCV'] = df['High'] / df['Open'] - 1
    
    # Calculate a factor 'factor_LO_OHLCV' based on the ratio of low to open prices
    df['factor_LO_OHLCV'] = df['Low'] / df['Open'] - 1
    
    # Calculate a factor 'factor_CH_OHLCV' based on the ratio of close to high prices
    df['factor_CH_OHLCV'] = df['Close'] / df['High'] - 1
    
    # Calculate a factor 'factor_CL_OHLCV' based on the ratio of close to low prices
    df['factor_CL_OHLCV'] = df['Close'] / df['Low'] - 1
    
    # Calculate a factor 'daily' based on the ratio of next day's close to next day's open
    df['daily'] = df['Close_t+1'] / df['Open_t+1'] - 1
    
    # Calculate a factor 'factor_amihud_highLow_OHLCV' based on the ratio of high-low range to volume
    df['factor_amihud_highLow_OHLCV'] = (df['factor_HL_OHLCV'] * 1000000) / df['Volume']
    
    # Create a new column 'Close_t-1' that contains the previous day's closing prices
    df['Close_t-1'] = df['Close'].shift(1)
    
    # Calculate a factor 'factor_daily_OHLCV' based on the ratio of current day's close to previous day's close
    df['factor_daily_OHLCV'] = df['Close'] / df['Close_t-1'] - 1
    
    # Calculate a factor 'factor_amihud_OHLCV' based on the ratio of daily price change to volume
    df['factor_amihud_OHLCV'] = (df['factor_daily_OHLCV'] * 1000000).abs() / df['Volume']
    
    # Commented out line - Calculate a factor 'factor_size_OHLCV' based on the product of trades and close price
    # df['factor_size_OHLCV'] = df['Trades'] * df['Close']

# Assuming that df is defined somewhere in the code
create_feature(df)

# Sort the DataFrame by 'Date' and 'Symbol', and drop rows with missing values
df = df.sort_values(['Date', 'Symbol']).dropna()

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame to include data from '2000-01-04' to '2021-04-29'
df = df[(df['Date'] >= '2000-01-04') & (df['Date'] <= '2021-04-29')]

# Group the DataFrame by 'Symbol' and count the number of rows (dates) for each stock
df.groupby('Symbol')['Date'].count()
#you can print this also

# Time series generation

'''
In this step, it is quite crucial because our data structure is panel data. If we build a time series model in the usual way, it will combine different stocks into one window. So, we need to create generators for each individual stock and then merge these generators into one large batch. The train-test split should also be done based on stocks.

You might wonder why not initially create separate generators for each stock, make predictions, and then select stocks afterward. This is because I want to build a shared model, which means the model can learn from other stocks' variations during training. The advantage of this approach is that it can capture industry-related patterns, increase the amount of data available, and reduce the risk of overfitting.
'''

# split data
def split_data(df, date, features, target):
    train = df[df['Date'] < date]
    test = df[df['Date'] >= date]
    X_train, y_train = np.array(train[features]), train[target].values
    X_test, y_test = np.array(test[features]), test[target].values
    return X_train, y_train, X_test, y_test

# check shape
def check_shape(data):
    print(data.shape)
    
check_shape(df)
# create time series
length = 60
batch_size = 64
train_generators = []
test_generators = []
stock_list = df['Symbol'].unique().tolist()
features = [i for i in df.columns if 'factor_' in i]

scalers = {}
for stock in stock_list:
    stock_data = df[df['Symbol'] == stock]
    X_train, y_train, X_test, y_test = split_data(stock_data, '2019-01-01', features, 'daily')
    scaler = MinMaxScaler((-1,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scalers[stock] = scaler
    train_gen = TimeseriesGenerator(X_train, y_train, length=length, batch_size=batch_size)
    test_gen = TimeseriesGenerator(X_test, y_test, length=length, batch_size=batch_size)
    train_generators.append(train_gen)
    test_generators.append(test_gen)

X_train_batches = []
y_train_batches = [] # len(X_train_batches) : 17、64、60、9

for gen in train_generators:
    X_train_batch, y_train_batch = next(iter(gen))
    X_train_batches.append(X_train_batch)
    y_train_batches.append(y_train_batch)

X_train_combines = np.concatenate(X_train_batches, axis=0) 
y_train_combines = np.concatenate(y_train_batches, axis=0)
print(f"X_combine shape:{X_train_combines.shape}") 
print(f"y_combine shape:{y_train_combines.shape}") 

X_test_batches = []
y_test_batches = []

for gen in test_generators:
    X_test_batch, y_test_batch = next(iter(gen))
    X_test_batches.append(X_test_batch)
    y_test_batches.append(y_test_batch)

X_test_combines = np.concatenate(X_test_batches, axis=0)
y_test_combines = np.concatenate(y_test_batches, axis=0)


#Building LSTM MODEL WITH XGBOOST
model = Sequential()
model.add(LSTM(50, activation='tanh',return_sequences=True, input_shape=(length, len(features))))
model.add(Dropout(0.1))
model.add(LSTM(25, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit the model
es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
model.fit(X_train_combines, y_train_combines, validation_data=(X_test_combines, y_test_combines), epochs=200, callbacks=[es, reduce_lr], verbose=0)

pd.DataFrame(model.history.history).plot()

test_pred = []
for stock_idx, gen in enumerate(test_generators):
    X_train_stock = train_generators[stock_idx].data
    current_batch = X_train_stock[-length:].reshape(1,length,len(features))
    
    stock_pred = []
    for i in range(len(gen.data)):
        current_pred = model.predict(current_batch, verbose=0)[0][0]
        stock_pred.append(current_pred)
        
        next_row = gen.data[i].reshape(1, 1, len(features))
        current_batch = np.append(current_batch[:, 1:, :], next_row, axis=1)
#     # inverse data with scaler if need
#     stock = stock_list[stock_idx]
#     stock_pred = scalers[stock].inverse_transform(np.array(stock_pred).reshape(-1,1)).flatten()
    test_pred.append(stock_pred)
test_pred = np.array(test_pred)

real_data = test_generators[1].targets
predicted_data = np.array(test_pred)[1]

plt.figure(figsize=(14, 7))
plt.plot(real_data, label="Real Data", color="blue")
plt.plot(predicted_data, label="Predicted Data", color="red", linestyle="--")
plt.title(f"Comparison of Real vs Predicted Data for Stock {stock_idx + 1}")
plt.xlabel("Days")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

actual_returns = []
for stock in stock_list:
    stock_data = df[(df['Symbol'] == stock) & (df['Date'] >= '2019-01-01')]
    actual_returns.append(stock_data['daily'].values)
actual_returns = np.array(actual_returns)

# choose the highest daily return stock
stock_choose = np.argmax(test_pred, axis=0) # (576, 1)
stock_choose_return = actual_returns[stock_choose, np.arange(actual_returns.shape[1])]

ret = pd.DataFrame(stock_choose_return, columns=['Return'])
ret['Return_fee'] = ret['Return'] - (2*0.001425*0.28) - 0.003
ret['Return_cumsum'] = ret['Return_fee'].cumsum()
ret['Return_cumprod'] = (1 + ret['Return_fee']).cumprod() - 1
ret['Return_cumprod_dd'] = [(ret['Return_fee'] + 1)[i]/(ret['Return_cumprod'] + 1)[0:i].max() - 1 for i in range(len(ret))]

print(ret)

ret['Return_cumprod'].plot()

single_period_return = ret['Return_cumsum'].iloc[-1]
compound_return = ret['Return_cumprod'].iloc[-1]
max_drawdown = ret['Return_cumprod_dd'].min()

print(single_period_return)
print(compound_return)
print(max_drawdown)

# with XGBoost LTR
dtrain = xgb.DMatrix(test_pred.T.reshape(-1,1), label=actual_returns.T.flatten())
group = [17] * 576
dtrain.set_group(group)

param = {
    'objective':'rank:pairwise',
    'eta':0.1,
    'gamma':0.7,
    'min_child_weight':.1,
    'max_depth':8,
        }
bst = xgb.train(param, dtrain, num_boost_round=4)
xgb_pred = bst.predict(dtrain)
xgb_pred_ranked = xgb_pred.reshape(576,17)

stock_choose_xgb = np.argmax(xgb_pred_ranked, axis=1)
stock_choose_xgb_return = actual_returns[stock_choose_xgb, np.arange(actual_returns.shape[1])]

ret_xgb = pd.DataFrame(stock_choose_xgb_return, columns=['Return'])
ret_xgb['Return_fee'] = ret_xgb['Return'] - (2*0.001425*0.28) - 0.003
ret_xgb['Return_cumsum'] = ret_xgb['Return_fee'].cumsum()
ret_xgb['Return_cumprod'] = (1 + ret_xgb['Return_fee']).cumprod() - 1
ret_xgb['Return_cumprod_dd'] = [(ret_xgb['Return_fee'] + 1)[i]/(ret_xgb['Return_cumprod'] + 1)[0:i].max() - 1 for i in range(len(ret_xgb))]

print(ret_xgb)

ret_xgb['Return_cumsum'].plot()

single_period_return = ret_xgb['Return_cumsum'].iloc[-1]
compound_return = ret_xgb['Return_cumprod'].iloc[-1]
max_drawdown = ret_xgb['Return_cumprod_dd'].min()

print(single_period_return)
print(compound_return)
print(max_drawdown)