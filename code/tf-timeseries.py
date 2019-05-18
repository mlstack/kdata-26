#! /c/Apps/Anaconda3/python

"""
[source] https://github.com/JustinBurg/TensorFlow_TimeSeries_RNN_MapR
"""

import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt
rng = pd.date_range(start='2000', periods=209, freq='M')
ts = pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()
# ts.plot(c='b', title='Time Series')
# plt.show()
print(ts.head(10))

TS = np.array(ts)
# print(TS)
num_periods = 20
f_horizon = 1
x_data = TS[:(len(TS)-(len(TS)%num_periods))]
# print(x_data)
x_batches = x_data.reshape(-1,20,1)

y_data = TS[1:(len(TS)-(len(TS)%num_periods))+f_horizon]
y_batches = y_data.reshape(-1,20,1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[:1])

print(len(y_batches))
print(y_batches.shape)
print(y_batches[:1])

def test_data(series, forecast, num_periods):
	test_x_setup = TS[-(num_periods+forecast):]
	testX = test_x_setup[:num_periods].reshape(-1,20,1)
	testY = TS[-(num_periods):].reshape(-1,20,1)
	return testX, testY

X_test, Y_test = test_data(TS, f_horizon,num_periods)
print(X_test.shape)

tf.reset_default_graph()

num_periods = 20
inputs = 1
hidden = 100
output = 1

x = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, inputs])

# cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, state = tf.nn.dynamic_rnn(cell, x, dtype= tf.float32)
stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

learning_rate = 0.001

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

epochs=2000
for ep in range(epochs):
	sess.run(
			training_op
		,	feed_dict = {
				x : x_batches
			,	y : y_batches
			}
		)
	if ep % 10 == 0:
		mse = loss.eval(
			feed_dict = {
				x : x_batches
			,	y : y_batches
			}
			)
		print(ep, '[mse]',mse)


y_pred = sess.run(
		outputs
	,	feed_dict = {
			x : X_test
		}
	)
print(y_pred)

plt.title('Actual & Forecast')
plt.plot(pd.Series(np.ravel(Y_test)),'b', markersize=10, label='Actual')
plt.plot(pd.Series(np.ravel(y_pred)),'r', markersize=10, label='Forecast')
plt.legend(loc='upper left')
plt.xlabel('Time Period')
plt.show()
