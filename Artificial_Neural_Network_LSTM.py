'''
    In this program I implemented a artificial neural network to predict the price of Apple stock -AAPL,
    of the 13th month using as values the last 12 month averages

'''

import math
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#from StockMarketPredictionbyMonth.Prediction_Months import predictionUpOrDown

plt.style.use('fivethirtyeight')

'''
    I have used as data the csv file 'MonthlyAverages', which was computed in the program 
    by calculating each the average of Adj Close Price in a whole month of transactions
'''

df = pd.read_csv('MonthlyAverage.csv')


# Lets now plot the price history over the last 12 months
plt.figure(figsize=(8,6))
plt.title('Average Monthly Price of AAPL for the last 12 months')
plt.plot(df['Avg Close Price'],color='blue' )
plt.xlabel('Date in Months', fontsize=11)
plt.ylabel('Average Price', fontsize=11)
plt.show()

# create the data set and make it a numpy array
data = df.filter(['Avg Close Price'])
dataset = data.values

# Get the number of rows to train the model on
lengthOfTrainingData = math.ceil(len(dataset) * .8)

# scale the new data
s = MinMaxScaler(feature_range=(0, 1))
dataScaled = s.fit_transform(dataset)


'''
    Create the training set
'''
train_data = dataScaled[1:lengthOfTrainingData, :]


# Split the data into x_train and y_train data sets
# create 2 lists
x_train = []
y_train = []

# Populate the x_train and the y_train
for i in range(2, len(train_data)):
    x_train.append(train_data[i - 1:i, 0])
    y_train.append(train_data[i, 0])


x_train = np.array(x_train)
y_train = np.array(y_train)

# Since the LSTM needs data input in a 3 dimensional form lets reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


'''
    Create the testing set
'''

test_data = dataScaled[lengthOfTrainingData - 1:, :]

x_test = []
y_test = dataset[lengthOfTrainingData:, :]
for i in range(1, len(test_data)):
    x_test.append(test_data[i - 1:i, 0])

x_test = np.array(x_test)

# Since the LSTM needs data input in a 3 dimensional form lets reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


'''
    Lets now create the Long Short Term Memory Model
'''

prediction_Model = Sequential()
prediction_Model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
prediction_Model.add(LSTM(50, return_sequences=False))
prediction_Model.add(Dense(25))
prediction_Model.add(Dense(1))

prediction_Model.compile(optimizer='adam', loss='mean_squared_error')

# Lets train the network model now using fit
prediction_Model.fit(x_train, y_train, batch_size=1, epochs=9, verbose=2)

# Get the models predictions
predictions = prediction_Model.predict(x_test)
predictions = s.inverse_transform(predictions)

# Calculate the RMSE -root mean squared errror
rootMeanSquaredError = np.sqrt(((predictions - y_test) ** 2).mean())



'''
    Now lets plot our data
'''
trainingData = data[:lengthOfTrainingData]
print("TRainni", trainingData)

pricePredictions = data[lengthOfTrainingData:]
print("REST", pricePredictions)
pricePredictions['Predictions'] = predictions

plt.figure(figsize=(8, 6))
plt.title('Long Short Term Memory Model on AAPL')
ax = plt.gca()
ax.set_facecolor('xkcd:black')


plt.xlabel('Date in months', fontsize=11)
plt.ylabel('Average Monthly Price', fontsize=11)

plt.plot(trainingData['Avg Close Price'], color='red')
plt.plot(pricePredictions[['Avg Close Price']], color='blue')
plt.plot(pricePredictions[['Predictions']], color='green')

plt.legend(['Data used in Training Set', 'Real Price Values', 'Predictions of Price'], loc='upper left')
plt.show()

# Print the predicted prices to see how they compare with actual ones
print("--------------------------------------------------------")
print("")
print("Predicted Prices: ", pricePredictions)
#
data = pd.read_csv('MonthlyAverage.csv')
avgClosePrice = data.filter(['Avg Close Price'])

lastMonth = avgClosePrice[-1:].values


# We can scale the values again just to make sure we are cautions and not using the real values

lastMonthScaled = s.transform(lastMonth)
X_test = [lastMonthScaled]

# make the  X_test data set a numpy array and again reshape it
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictedPrice = prediction_Model.predict(X_test)

Price = s.inverse_transform(predictedPrice)

print("--------------------------------------------------------")
print("")
print("Predicted Price of the next month is: ")
print("---> using LSTM model: $", Price)
print("----------------------------------------------------------")
print("")
print("The LSTM Model has a RMSE = :")
print("Root Mean Squared Error is: ", (round)((rootMeanSquaredError / 100), 3), " %")

#print(predictionUpOrDown())
