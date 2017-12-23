#!/usr/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# number of features
INPUT_SIZE = 19

# opens the csv file with the stock data.
# creates train-test split of the stock data.
def getData():
	file = 'INTC.csv'
	data = pd.read_csv(file)
	X = data.Close[:9500]
	X = X.values.reshape((475, INPUT_SIZE+1))
	y = X[:, -1]
	X = X[:, :INPUT_SIZE]

	return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
	# gets the data after the split
	# the X data contains stock prices of 19 consecutive days
	# the y data is the stock price of the 20th day
	X_train, X_test, y_train, y_test = getData()
	print 'Train Size: ' + str(X_train.shape)
	print 'Test Size: ' + str(X_test.shape)

	learning_rate = tf.constant(0.0001)

	# layers: 19 -> 16 -> 4 -> 1
	input_size = INPUT_SIZE
	hidden1_size = 16
	hidden2_size = 4
	output_size = 1

	# placeholder for input and output
	X = tf.placeholder(tf.float32, shape=[None, input_size])
	y = tf.placeholder(tf.float32, shape=[None])

	# the weights and biases of the layers
	weights = {
		'h1': tf.Variable(tf.truncated_normal([input_size, hidden1_size], stddev=0.1)),
		'h2': tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1)),
		'hout': tf.Variable(tf.truncated_normal([hidden2_size, output_size], stddev=0.1))
	}
	biases = {
		'b1': tf.Variable(tf.constant(0.1, shape=[hidden1_size])),
		'b2': tf.Variable(tf.constant(0.1, shape=[hidden2_size])),
		'bout': tf.Variable(0.)
	}

	# the creation of the layers using relu activation
	hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
	hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2']))
	out_layer = tf.add(tf.matmul(hidden_layer2, weights['hout']), biases['bout'])

	out_layer = tf.transpose(out_layer)
	loss = tf.reduce_mean(tf.squared_difference(out_layer, y))
	update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# start learning
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for i in range(11000):
		_, loss_ = sess.run([update, loss], feed_dict={X: X_train, y: y_train})

		# prints the loss of the training and testing data each 200 epochs
		if i % 200 == 0:
			testErr = sess.run(loss, feed_dict={X: X_test, y: y_test})
			print(' '.join(['Training loss:', str(loss_), 'Test loss:', str(testErr)]))

	# prints the results
	trainErr = sess.run(loss, feed_dict={X: X_train, y: y_train})
	print 'Train Mean Squared Error: ' + str(trainErr)
	testErr = sess.run(loss, feed_dict={X: X_test, y: y_test})
	print 'Test Mean Squared Error: ' + str(testErr)
	sess.close()