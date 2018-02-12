# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:29:51 2018

@author: trevo
"""

"""
Demostrate ML with a linear regression algorithm built in tensorflow
with a L2 loss function and gradient descent optimizer for batch and
stochastic training 

Find a weight value A from a x dataset feature with a y target value
"""
# import numpy and tensorflow modules
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

# Establish a tensorflow graph session
sess = tf.Session()

# Define batch size
batch_size = 20

# Define dataset and target values
x_vals = np.random.normal(1,0.1, 100)
y_vals = np.repeat(10.,100)

# Establish tensorflow 2D placeholders for graph session data feeding
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],  dtype=tf.float32)

# Define the variable A weight for the tensorflow graph session to learn and modified
# Initialize weight A with a random value 
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Define the graph matrix operation
my_output = tf.matmul(x_data, A)

# Define the mean l2 loss function over each batch
loss = tf.reduce_mean(tf.square(my_output - y_target))

# Initialize variable for graphic session
init = tf.global_variables_initializer()
sess.run(init)

# Define gradient descent optimization algorithm for graphic session
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

# Initialize loss batch list
loss_batch_list = []
print('Batch Training, Size=20')
# Define number of sessions for training
sess_num = 100
# loop for batch graphic training sessions
for i in range(0,sess_num):
    # Generate random number between 0 to 99 for training data selection
    rand_index = np.random.choice(100, size=batch_size)
    # Select random x feature value and y target value for training
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    # run graphic training session with feed placeholder data
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    # Display algorithm status after each block on 25 iterations
    if ((i+1)%5 == 0):
        print('Step#{} A = {}'.format((i+1),sess.run(A)))
        temp_loss = sess.run(loss,\
              feed_dict={x_data:rand_x, y_target:rand_y})
        print('Loss = {}'.format(temp_loss))
        loss_batch_list.append(temp_loss)
    
print()
# Reset graph and reintialize the graph session for stochastic training
ops.reset_default_graph()
sess = tf.Session()

# Define dataset and target values
x_vals = np.random.normal(1,0.1, 100)
y_vals = np.repeat(10.,100)

# Establish tensorflow placeholders for graph session data feeding
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Define the variable A weight for the tensorflow graph session to learn and modified
# Initialize weight A with a random value 
A = tf.Variable(tf.random_normal(shape=[1]))

# Define the graph operation
my_output = tf.multiply(x_data, A)

# Define the loss function, L2
loss = tf.square(my_output - y_target)

# Initialize variable for graphic session
init = tf.global_variables_initializer()
sess.run(init)

# Define gradient descent optimization algorithm for graphic session
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

# loop to graphic training sessions
loss_stochastic_list = []
print('Stochastic Training')
# Define number of sessions for training
sess_num = 100
# loop for stochastic graphic training sessions
for i in range(0,sess_num):
    # Generate random number between 0 to 99 for training data selection
    rand_index = np.random.choice(100)
    # Select random x feature value and y target value for training
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    # run graphic training session with feed placeholder data
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    # Display algorithm status after each block on 25 iterations
    if ((i+1)%5 == 0):
        print('Step#{} A = {}'.format((i+1),sess.run(A)))
        temp_loss = sess.run(loss,\
              feed_dict={x_data:rand_x, y_target:rand_y})
        print('Loss = {}'.format(temp_loss))
        loss_stochastic_list.append(temp_loss)
        
# Plot the batch and the stochastic losses
plt.plot(range(0, 100, 5),loss_stochastic_list, 'b-', label = 'Stochastic Loss')
plt.plot(range(0, 100, 5),loss_batch_list, 'r--', label = 'Batch Loss, size 20')
plt.legend(loc='upper right', prop={'size':11})
plt.show()
