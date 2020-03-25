import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from scipy import stats
import wx
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
plt.style.use('fivethirtyeight')

"""
    First lets create the average of the months. 
    It will be created calculating the Adj Close and dividing it at the numbers of dats
"""

# Calculate average of months
march = pd.read_csv('AAPL_March.csv')
april = pd.read_csv('AAPL_April.csv')
may = pd.read_csv('AAPL_May.csv')

june = pd.read_csv('AAPL_June.csv')
july = pd.read_csv('AAPL_July.csv')
august = pd.read_csv('AAPL_August.csv')

september = pd.read_csv('AAPL_September.csv')
october = pd.read_csv('AAPL_October.csv')
november = pd.read_csv('AAPL_November.csv')

december = pd.read_csv('AAPL_December.csv')
january = pd.read_csv('AAPL_January2020.csv')
february = pd.read_csv('AAPL_February2020.csv')

# -------------------------------

data = pd.read_csv('MonthlyAverage.csv')
df = np.array(data['Avg Close Price'])


def normalizeData(dataFrame):
    myData = pd.read_csv('MonthlyAverage.csv')
    df = np.array(myData['Avg Close Price'])

    # Normalize the data using z-score
    normalizedData = stats.zscore(df)

    return normalizedData


'''
# --------------------------------------
'''


def calculateAverage(myData):
    # Get all of the rows from the Close column
    closePrice = np.array(myData['Adj Close'])

    numOfDays = (int(len(closePrice)))
    total = 0

    for x in range(0, numOfDays):
        total = total + closePrice[x]

    average = total / numOfDays

    return average


avgMarch = calculateAverage(march)
avgApril = calculateAverage(april)
avgMay = calculateAverage(may)

avgJune = calculateAverage(june)
avgJuly = calculateAverage(july)
avgAugust = calculateAverage(august)

avgSeptember = calculateAverage(september)
avgOctober = calculateAverage(october)
avgNovember = calculateAverage(november)

avgDecember = calculateAverage(december)
avgJanuary = calculateAverage(january)
avgFebruary = calculateAverage(february)

print(" -------------------------------------------------------------------------------")
print(" ")
print("Values of monthly averages: ")
print(" ")
print("March: ", avgMarch, " ", "April: ", avgApril, " ", "May: ", avgMay, " ")
print("June: ", avgJune, " ", "July: ", avgJuly, " ", "August: ", avgAugust, " ")
print("September: ", avgSeptember, " ", "October: ", avgOctober, " ", "November ", avgNovember, " ")
print("December: ", avgDecember, " ", "January: ", avgJanuary, " ", "February: ", avgFebruary)
print("--------------------------------------------------------------------------------")
print(" ")

'''

# This is a function to calculate the prediction of the 13th month
# We will be doing it by calculating the average of the total months averages and
# - if the prediction is higher than that it goes Up or else Down

'''


def predictionUpOrDown():
    monthlyAverageData = pd.read_csv('MonthlyAverage.csv')
    closePrice = np.array(monthlyAverageData['Avg Close Price'])

    numMonths = (int(len(monthlyAverageData)))
    avgY = 0

    for x in range(0, numMonths):
        avgY = avgY + closePrice[x]

    averageYear = avgY / numMonths

    return averageYear


print("Value of the whole year average: ")
print(" ")
a = predictionUpOrDown()
print(a)
print("--------------------------------------------------------------------------------")
print(" ")


###################
# Lets now plot the price history over the last 12 months
plt.figure(figsize=(8,6))
plt.title('Average Monthly Price of AAPL for the last 12 months')
plt.plot(data['Avg Close Price'])
plt.xlabel('Date in Months', fontsize=11)
plt.ylabel('Average Price', fontsize=11)
plt.show()
'''
# We, now have to read the new file of monthly averages and create and implement the Support vector machines
# plot the results of each kernel and then make the comparison to se if the 13th month is going up or down

'''

monthlyAverage = pd.read_csv('MonthlyAverage.csv')
monthlyAverage.head(7)

# I want to create the x and y (date and average close price)
# Put them in lists for now
datesByMonth = []
pricesByMonth = []

# Get all of the data except for the last row
monthlyAveragePrice = monthlyAverage.head(len(monthlyAverage) - 1)
# print(monthlyAveragePrice.shape)

# Get all of the rows from the Date column
monthlyAverage_dates = monthlyAveragePrice.loc[:, 'Date']
# Get all of the rows from the Avg Close Price column
monthlyAverage_ClosePrice = monthlyAveragePrice.loc[:, 'Avg Close Price']

# Create the independent data set 'x' as dates
for date in monthlyAverage_dates:
    # I have to separate it by months
    datesByMonth.append([int(date.split('-')[1])])

# Create the dependent data set 'y' as close prices of the months
for open_price in monthlyAverage_ClosePrice:
    pricesByMonth.append(float(open_price))

# Print the dates by months
print("The months that we are getting the data from are: ", datesByMonth)
print(" ")

# Create a variable named forecast which will be our prediction.
# It will be set up to 1 (predict one month in advance)
forecast = 1

'''
# Now we will create 3 functions to make predictions using 3 different support vector regression models
# with 3 different kernels = radial basis function, linear and polynomial
'''


def predictAveragePriceRBF(date, averagePrice, forecast):
    # Create Support Vector Regression Model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Train the model on the dates and average prices
    svr_rbf.fit(date, averagePrice)

    # Plot the model on a graph to see which has the best fit
    plt.scatter(date, averagePrice, color='black', label='Data')

    plt.plot(date, svr_rbf.predict(date), color='red', label='RBF model')

    plt.xlabel('Date by Months')
    plt.ylabel('Average Price by Months')

    plt.title('Support Vector Machine - SVM')
    plt.legend()
    plt.show()

    # return the model prediction
    return svr_rbf.predict(forecast)[0]


def predictAveragePriceRegression(date, price, forecast):
    quadratic_Regression = make_pipeline(PolynomialFeatures(2), Ridge())
    quadratic_Regression.fit(date, price)

    # Plot the model on a graph to see which has the best fit
    plt.scatter(date, price, color='black', label='Data')

    plt.plot(date, quadratic_Regression.predict(date), color='yellow', label='Regression model')

    plt.xlabel('Date by Months')
    plt.ylabel('Average Price by Months')

    plt.title('Quadratic regression Model')
    plt.legend()
    plt.show()

    return quadratic_Regression.predict(forecast)


def getRegressionAccuracy():
    normalized = pd.read_csv('NormalizedData.csv')

    months = normalized[['Avg Close Price']]
    months['Prediction'] = months[['Avg Close Price']].shift(-forecast)

    # Create train set as the price per month
    train = np.array(months.drop(['Prediction'], 1))
    train = train[:-forecast]

    # Create test set as the column prediction
    test = np.array(months['Prediction'])
    test = test[:-forecast]

    # Split the data
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        test, test_size=0.2, random_state=109)

    quadratic_Regression = make_pipeline(PolynomialFeatures(2), Ridge())
    quadratic_Regression.fit(X_train, y_train)

    # Printing the results as the confidence level
    return quadratic_Regression.score(X_test, y_test)


# SVM - kernel poly
def predictAveragePricePoly(date, averagePrice, forecast):
    # Create Support Vector Regression Model
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    # Train the model on the dates and average prices
    svr_poly.fit(date, averagePrice)

    # Plot the model on a graph to see which has the best fit
    plt.scatter(date, averagePrice, color='black', label='Data')

    plt.plot(date, svr_poly.predict(date), color='blue', label='Polynomial model')

    plt.xlabel('Date by Months')
    plt.ylabel('Average Price by Months')

    plt.title('Support Vector Machine - SVM')
    plt.legend()
    plt.show()

    # return the model prediction
    return svr_poly.predict(forecast)[0]


# SVM linear
def predictAveragePriceLinear(date, averagePrice, forecast):
    # Create Support Vector Regression Model
    svr_lin = SVR(kernel='linear', C=1e3)

    # Train the model on the dates and average prices
    svr_lin.fit(date, averagePrice)

    # Plot the model on a graph to see which has the best fit
    plt.scatter(date, averagePrice, color='black', label='Data')

    plt.plot(date, svr_lin.predict(date), color='green', label='Linear model')

    plt.xlabel('Date by Months')
    plt.ylabel('Average Price by Months')

    plt.title('Support Vector Machine - SVM')
    plt.legend()
    plt.show()

    # return the model prediction
    return svr_lin.predict(forecast)[0]


predicted_priceRBF = predictAveragePriceRBF(datesByMonth, pricesByMonth, [[13]])
predicted_priceLinear = predictAveragePriceLinear(datesByMonth, pricesByMonth, [[13]])
predicted_pricePoly = predictAveragePricePoly(datesByMonth, pricesByMonth, [[13]])

'''
# Creating the SVM to get the accuracy of the models using different kernels

'''


# ------ Get the LINEAR model accuracy ---------------------------------------------
def getAccuracyLINEAR():
    normalized = pd.read_csv('NormalizedData.csv')

    months = normalized[['Avg Close Price']]
    months['Prediction'] = months[['Avg Close Price']].shift(-forecast)

    # Create train set as the price per month
    train = np.array(months.drop(['Prediction'], 1))
    train = train[:-forecast]

    # Create test set as the column prediction
    test = np.array(months['Prediction'])
    test = test[:-forecast]

    # Split the data
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        test, test_size=0.2, random_state=109)

    lin = SVR(kernel='linear', C=1e3)
    lin.fit(X_train, y_train)

    return lin.score(X_test, y_test)


# ------ Get the RBF model accuracy ---------------------------------------------
def getAccuracyRBF():
    normalized = pd.read_csv('NormalizedData.csv')

    months = normalized[['Avg Close Price']]
    months['Prediction'] = months[['Avg Close Price']].shift(-forecast)

    # Create train set as the price per month
    train = np.array(months.drop(['Prediction'], 1))
    train = train[:-forecast]

    # Create test set as the column prediction
    test = np.array(months['Prediction'])
    test = test[:-forecast]

    # Split the data
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        test, test_size=0.2, random_state=109)

    rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    rbf.fit(X_train, y_train)

    return rbf.score(X_test, y_test)


# ------ Get the POLYNOMIAL model accuracy ---------------------------------------------
def getAccuracyPOLY():
    months = monthlyAverage[['Avg Close Price']]
    months['Prediction'] = months[['Avg Close Price']].shift(-forecast)

    # Create train set as the price per month
    train = np.array(months.drop(['Prediction'], 1))
    train = train[:-forecast]

    # Create test set as the column prediction
    test = np.array(months['Prediction'])
    test = test[:-forecast]

    # Split the data
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        test, test_size=0.2, random_state=109)

    poly = SVR(kernel='poly', C=1e3, degree=2)
    poly.fit(X_train, y_train)

    return poly.score(X_test, y_test)


'''
    Function to implement from skcit learn library the KNN algorithm and train it on our data
    Also will plot the result separately and all together to see the differences between the models
'''


def makePredKNN(date, averagePrice, forecast):
    # Create the KNN
    k_NN = KNeighborsRegressor(n_neighbors=3)
    k_NN.fit(date, averagePrice)

    price = k_NN.predict(forecast)

    # Plot the model on a graph to see which has the best fit
    plt.scatter(date, averagePrice, color='black', label='Data')

    plt.plot(date, k_NN.predict(date), color='purple', label='K-NN Model')

    plt.xlabel('Date by Months')
    plt.ylabel('Average Price by Months')

    plt.title('K - Nearest Neighbour')
    plt.legend()
    plt.show()

    # return the model prediction
    return price


# Function to get the KNN model acuuracy
def getAccuracyKNN():
    normalized = pd.read_csv('NormalizedData.csv')

    months = normalized[['Avg Close Price']]
    months['Prediction'] = months[['Avg Close Price']].shift(-forecast)

    # Create train set as the price per month
    train = np.array(months.drop(['Prediction'], 1))
    train = train[:-forecast]

    # Create test set as the column prediction
    test = np.array(months['Prediction'])
    test = test[:-forecast]

    # Split the data
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        test, test_size=0.2, random_state=109)

    k_NN = KNeighborsRegressor(n_neighbors=3)
    k_NN.fit(X_train, y_train)

    return k_NN.score(X_test, y_test)


predicted_PriceKNN = makePredKNN(datesByMonth, pricesByMonth, [[13]])
predicted_PriceRegression = predictAveragePriceRegression(datesByMonth, pricesByMonth, [[13]])

print("--------------------------------------------------------------------------")
print(" ")
print("The predicted price of the next month is: ")
print("")

print("---> using RBF kernel: $", predicted_priceRBF)
print("-> the model has a accuracy of: ",
      round((getAccuracyRBF() * 100), 3), "%")
print("")
print("---> using Polynomial kernel: ", predicted_pricePoly)
print("-> the model has a accuracy of: ",
      round((getAccuracyPOLY() * 100), 3), "%")
print("")
print("---> using Linear kernel: ", predicted_priceLinear)
print("-> the model has a accuracy of: ",
      round((getAccuracyLINEAR() * 100), 3), "%")

print(" ")
print("---> using KNN model: ", predicted_PriceKNN)
print("-> the model has a accuracy of: ",
      round((getAccuracyKNN() * 100), 3), "%")

print("")
print("---> using Regression model : ", predicted_priceLinear)
print("-> the model has a accuracy of: ",
      round((getRegressionAccuracy() * 100), 3), "%")
print("--------------------------------------------------------------------------")


def bestAccuracy():
    rbf = getAccuracyRBF()
    lin = getAccuracyLINEAR()
    poly = getAccuracyPOLY()
    knn = getAccuracyKNN()
    reg = getRegressionAccuracy()

    maxList = [rbf, lin, poly, knn, reg]

    return max(maxList)


# Function to get the best price(given by the model with the best accuracy)
def getBestPrice():
    rbf = getAccuracyRBF()
    lin = getAccuracyLINEAR()
    poly = getAccuracyPOLY()
    knn = getAccuracyKNN()
    reg = getRegressionAccuracy()

    accuracies = {rbf: predicted_priceRBF,
                  lin: predicted_priceLinear,
                  poly: predicted_pricePoly,
                  knn: predicted_PriceKNN,
                  reg: predicted_PriceRegression
                  }

    if bestAccuracy() == rbf:
        return accuracies[rbf]
    elif bestAccuracy() == lin:
        return accuracies[lin]
    elif bestAccuracy() == knn:
        return accuracies[knn]
    elif bestAccuracy() == reg:
        return accuracies[reg]
    else:
        return accuracies[poly]

    return predictedPrice


def getPrice2Decimals(bestPrice):
    return "{:0.2f}".format(bestPrice)


print(getPrice2Decimals(getBestPrice()))


# Function to make to make prediction if the month is going up or down
def makePred(pricePredicted, yearAverage):
    if getBestPrice() < predictionUpOrDown():
        print("The price of the next month will be: $", getBestPrice())
        print("")
        print("The predicted month will go DOWN. ")
        print("You should NOT buy the stock now")
    elif getBestPrice() > predictionUpOrDown():
        print("The price of the next month will be: ", getBestPrice())
        print("")
        print("The predicted month will go UP. ")
        print("You SHOULD buy the stock now")
    else:
        print("The stock price will keep the same value")

    print(" ")


def plotAllModels(date, averagePrice, forecast):
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_poly.fit(date, averagePrice)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_lin.fit(date, averagePrice)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(date, averagePrice)
    k_NN = KNeighborsRegressor(n_neighbors=3)
    k_NN.fit(date, averagePrice)
    quadratic_Regression = make_pipeline(PolynomialFeatures(2), Ridge())
    quadratic_Regression.fit(date, averagePrice)

    # Plot the model on a graph to see which has the best fit
    plt.scatter(date, averagePrice, color='black', label='Data')
    plt.plot(date, svr_lin.predict(date), color='green', label='Linear model')
    plt.plot(date, svr_poly.predict(date), color='blue', label='Polynomial model')
    plt.plot(date, svr_rbf.predict(date), color='red', label='RBF model')
    plt.plot(date, k_NN.predict(date), color='purple', label='KNN model')
    #plt.plot(date, quadratic_Regression.predict(date), color='yellow', label="Regression model")

    plt.xlabel('Date by Months')
    plt.ylabel('Average Price by Months')

    plt.title('PREDICTION MODELS')
    plt.legend()
    plt.show()


print("")
print("My Predictions for the next month price stock are: ")
print(" ")
print(makePred(getBestPrice(), a))
print("")
print("--------------------------------------------------------------------------")
print(" ")
plotAllModels(datesByMonth, pricesByMonth, 13)
