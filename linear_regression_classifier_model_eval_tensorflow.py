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
# import numpy, tensorflow, and matplotlib.pyplot modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

# Reset graph and graph session
ops.reset_default_graph()

# Establish a tensorflow graph session
sess = tf.Session()

# Define dataset and target values
# np.random.normal(mean, std, size)
x_vals =\
 np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
y_vals = np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))

# Establish tensorflow placeholders for graph session data feeding
x_data = tf.placeholder(shape=[1, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)

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

# Define the variable A weight for the tensorflow graph session to 
# learn and modify
# Initialize weight A with a random value 
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Define the graph operation for learn bias value A
my_output = tf.add(x_data, A)

# Define the loss function using unscaled logits with cross entropy
# Must define arguments with names
loss_xentropy =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits\
                              (labels = y_target, logits = my_output))

# Initialize variable for graphic session
init = tf.global_variables_initializer()
sess.run(init)

# Define gradient descent optimization algorithm for graphic session
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = my_opt.minimize(loss_xentropy)

# loop to graphic training sessions
# Define number of sessions for training
sess_num = 1800
for i in range(0,sess_num):
    # Generate random numbers based on training data and batch size
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    # Select random x feature value and y target value for training
    # numpy transpose not required to match placeholder shape
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    # run graphic training session with feed placeholder data
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    # Display algorithm status after each block on 25 iterations
    if ((i+1)%200 == 0):
        print('Step#{} A = {}'.format((i+1),sess.run(A)))
        print('Loss = {}'.format(sess.run(loss_xentropy,\
              feed_dict={x_data:rand_x, y_target:rand_y})))

# Model Evaluation with prediction accuracy
# Compute target predictions from the training model outcome as the argument 
# to the sigmoid function.  Note: squeeze method remove single dimensions
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))

# Establish function to compare the model predictions with the target 
# classifications with True and False boolean values
correct_prediction = tf.equal(y_prediction, y_target)

# Convert boolean comparsion results to floating point data type
prediction_cast = tf.cast(correct_prediction,tf.float32)

# Define function to compute the prediction accuracy using mean tensor function
accuracy = tf.reduce_mean(prediction_cast)

acc_value_test = sess.run(accuracy, feed_dict={x_data:[x_vals_test],\
                                     y_target:[y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data:[x_vals_train],\
                                     y_target:[y_vals_train]})       
print('Accuracy on test dataset:{}'.format(acc_value_test))
print('Accuracy on train dataset:{}'.format(acc_value_train))      
# print(sess.run(prediction_cast, feed_dict={x_data:[x_vals_test],\
                                           # y_target:[y_vals_test]}))              
# print(sess.run(correct_prediction, feed_dict={x_data:[x_vals_test],\
                                           # y_target:[y_vals_test]})) 

#Visualize classifier accuracy in histogram

# Set variable to A bias value        
A_result = sess.run(A)

# Define x-axis values between negative and position five (5)
bins = np.linspace(-5, 5, num=50)

plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color = 'blue')
plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color = 'red')
# Plot bias value A
plt.plot((A_result,A_result), (0,8), 'k--',\
         linewidth=3, label='A = {}'.format(np.round(A_result,2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy = {}'.format(np.round(acc_value_test,2)))
plt.show()