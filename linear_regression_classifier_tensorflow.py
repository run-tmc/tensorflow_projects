# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:29:51 2018

@author: trevo
"""

"""
Demostrate ML with a linear regression classifier algorithm built in tensorflow
with a cross entropy loss function by transforming the logits value with a 
sigmoid and gradient descent optimizer 

Find a bias value A from a x dataset feature with a y target classification
"""
# import numpy and tensorflow modules
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

# Reset graph and graph session
ops.reset_default_graph()

# Establish a tensorflow graph session
sess = tf.Session()

# Define dataset and target values
# np.random.normal(mean, std, size)
x_vals =\
 np.concatenate((np.random.normal(-1,1, 50), np.random.normal(3,1, 50)))
y_vals = np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))

# Establish tensorflow placeholders for graph session data feeding
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Define the variable A weight for the tensorflow graph session to 
# learn and modify
# Initialize weight A with a random value 
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Define the graph operation for learn bias value A
my_output = tf.add(x_data, A)

# Add expanded dimensions for output and target variables
# to accommondate the lose function
my_output_expanded = tf.expand_dims(my_output,0)
y_target_expanded = tf.expand_dims(y_target,0)

# Define the loss function using unscaled logits with cross entropy
# Must define arguments with names
loss_xentropy =\
 tf.nn.sigmoid_cross_entropy_with_logits(labels = y_target_expanded,\
                                         logits = my_output_expanded)

# Initialize variable for graphic session
init = tf.global_variables_initializer()
sess.run(init)

# Define gradient descent optimization algorithm for graphic session
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = my_opt.minimize(loss_xentropy)

# loop to graphic training sessions
# Define number of sessions for training
sess_num = 1400
for i in range(0,sess_num):
    # Generate random number between 0 to 99 for training data selection
    rand_index = np.random.choice(100)
    # Select random x feature value and y target value for training
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    # run graphic training session with feed placeholder data
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    # Display algorithm status after each block of 200 iterations
    if ((i+1)%200 == 0):
        print('Step#{} A = {}'.format((i+1),sess.run(A)))
        print('Loss = {}'.format(sess.run(loss_xentropy,\
              feed_dict={x_data:rand_x, y_target:rand_y})))
    
              
    
