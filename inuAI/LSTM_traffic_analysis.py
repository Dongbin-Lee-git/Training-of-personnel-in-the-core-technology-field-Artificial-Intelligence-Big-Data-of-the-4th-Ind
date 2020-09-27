import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

# read cvs data
dff = pd. read_csv('P3_Input_CI.csv')

# extract the column from Dataframe
dff = dff['CI_seoul_in']
dff = pd.DataFrame(dff)
dff.columns = ['Ci']

# since it is Time series prediction
# train test split 80:20 ratio
train = dff['Ci'][0:13825]
test = dff['Ci'][13825:17280]

# plot input data
plt.figure(figsize=(18, 8))
plt.plot(dff)
ticks = [i*1440 for i in range(0, 13)]
labels = [str(i) for i in range(1, 14)]
plt.xticks(ticks, labels)
plt.xlabel('Day')
plt.ylabel('Congestion Index')
plt.grid()
plt.show()

# number of data in one Input sample
n_step = 120
x_train, y_train = [], []

# create the sequence of Input and output for Train data
for i in range(n_step, len(train)-20):
    x_train.append(train[i-n_step:i])
    y_train.append(train[i+20])

# Reshape to 3D to work with LSTM model
# number of sample, total element in one sample, 1 ==> order
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=15, batch_size=100, verbose=1)

# Create of Input sequence for Test data
X_test = []
y_test = []
train

for i in range(n_step, len(test)):
    X_test.append(test[i-n_step:i])

# since the first predicted value will be 140th data in test dataset
y_test = test[140:]

# convert into array
X_test = np.array(X_test)

# reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# prediction of Input of Test data set
CI_per = model.predict(X_test)
ci = CI_per.reshape(CI_per.shape[0])
y_test = np.array(y_test)

# Calculating Error
mse = (np.mean(np.power((y_test - CI_per), 2)))
rms = np.sqrt(np.mean(np.power((y_test - CI_per), 2)))
print(mse)
print(rms)

# adding new column into df DataFrame
df = dff.copy()
df['Pred'] = 0

# to add Predicted Value in df DataFrame with data Crossponding to its actual test data
index = np.array([i for i in range(13965, 13965+len(CI_per))])
df1 = pd.DataFrame(CI_per)
df1 = df1.set_index(index)
df1.columns = ['Pred']
df = df.append(df1)

# df = df.sort_values(by='Pred')
df = df.dropna(subset=['Pred'])

# train test and prediction value from data_all with index position
train1 = df['Ci'][:13825]
test1 = df['Ci'][13825:17280]
pred = df['Pred'][17280:]
plt.figure(figsize=(16, 8))

# plot function
plt.plot(train1, label='train')
plt.plot(test1, label='actual_test')
plt.plot(pred, label='LSTM_predicted_test')

# position to place the label in X-axis
ticks = [i*1440 for i in range(0, 13)]

# name of label
labels = [str(i) for i in range(0, 13)]
plt.xticks(ticks, labels)
plt.grid()
plt.title('LSTM model Prediction')
plt.xlabel('Day')
plt.ylabel('Congestion Index')
plt.legend()
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(test1, label='actual_test data')
plt.plot(pred, label='LSTM_predicted data')
ticks = [i*1440 for i in range(10, 13)]
labels = [str(i) for i in range(10, 13)]
plt.xticks(ticks, labels)
plt.grid()
plt.title('LSTM Prediction model')
plt.xlabel('Day')
plt.ylabel('Congestion Index')
plt.legend()
plt.show()

# position to place the label in  X-axis
ticks = [i*1440 for i in range(0, 13)]

# name of label
labels = [str(i) for i in range(0, 13)]
plt.xticks(ticks, labels)
plt.grid()
plt.title('LSTM model Prediction')
plt.xlabel('Day')
plt.ylabel('Congestion Index')
plt.legend()
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(test1, label='actual_test data')
plt.plot(pred, label='LSTM_predicted data')
ticks = [i*1440 for i in range(10, 13)]
labels = [str(i) for i in range(10, 13)]
plt.xticks(ticks, labels)
plt.grid()
plt.title('LSTM Prediction model')
plt.xlabel('Day')
plt.ylabel('Congestion Index')
plt.legend()
plt.show()