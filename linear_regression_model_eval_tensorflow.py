# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:29:51 2018

@author: trevo
"""

"""
Demostrate ML with a linear regression algorithm built in tensorflow
with a L2 loss function and gradient descent optimizer 

Find a weight value A from a x dataset feature with a y target value
Split dataset for model training and testing
Evaluate model with mean square error (MSE)

"""
# import numpy and tensorflow modules

import numpy as np
import tensorflow as tf

# Establish a tensorflow graph session
sess = tf.Session()

# Define dataset and target values
x_vals = np.random.normal(1,0.1, 100)
y_vals = np.repeat(10.,100)

# Establish tensorflow placeholders for graph session data feeding
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Define batch size
batch_size = 25

#Establish training set from random indices to be 80% of the feature size
# with no duplicates (replacements)
train_indices = np.random.choice(len(x_vals),\
                                 round(len(x_vals)*0.8), replace=False)

# Establish test set from remaining dataset indices
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

# Establish training feature dataset and target values
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]

# Establish test feature dataset and target values
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

# Define the variable A weight for the tensorflow graph session to learn and modified
# Initialize weight A with a random value 
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Define the graph operation
my_output = tf.matmul(x_data, A)

# Define the loss function, L2
loss = tf.reduce_mean(tf.square(my_output - y_target))

# Initialize variable for graphic session
init = tf.global_variables_initializer()
sess.run(init)

# Define gradient descent optimization algorithm for graphic session
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

# loop to graphic training sessions
# Define number of sessions for training
sess_num = 100
for i in range(0,sess_num):
    # Generate random numbers based on training data and batch size
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    # Select random x feature value and y target value for training
    # requires numpy transpose to match placeholder shape
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    # run graphic training session with feed placeholder data
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    # Display algorithm status after each block on 25 iterations
    if ((i+1)%25 == 0):
        print('Step#{} A = {}'.format((i+1),sess.run(A)))
        print('Loss = {}'.format(sess.run(loss,\
              feed_dict={x_data:rand_x, y_target:rand_y})))

# Model Evaluation with MSE
mse_test = sess.run(loss, feed_dict={x_data:np.transpose([x_vals_test]),\
                                     y_target:np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data:np.transpose([x_vals_train]),\
                                     y_target:np.transpose([y_vals_train])})       
print('MSE on test dataset:{}'.format(np.round(mse_test,2)))
print('MSE on train dataset:{}'.format(np.round(mse_train,2)))     
              
    
