# Stock-Multilayer-perceptron

This project was made has part of a learning project, dont use it for trading.

I picked Intel's historical data from Yahoo! Finance (9500 days of close prices).
The script uses the close prices of 19 consecutive days to predict the close price of the 21th day.
I used Multilayer-perceptron algorithm and Tensorflow to learn, test and predict.
I used the pandas’ library to read the csv file with the stock information, and to create the numpy arrays.

I performed a train-test split (80%-20%) using Sklearn (train_test_split) (380 for train, 95 for test)

Results:
Train Mean Squared Error: 0.238765
Test Mean Squared Error: 0.251052
