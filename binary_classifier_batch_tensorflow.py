# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:29:51 2018

@author: trevo
"""

"""
Demostrate ML with a binary classifier algorithm built in tensorflow
using a L2 loss function and gradient descent optimizer with batch training 

Find weight and bias values for classifying Iris dataset
"""
# import numpy and tensorflow modules
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import ops
from sklearn import datasets
import matplotlib.pyplot as plt

#load Iris dataset
iris = datasets.load_iris()

# Establish target array for setosa specie 
binary_target = np.array([1. if x==0 else 0. for x in iris.target])

# Establish 2D feature arrary with petal lengths and widths
iris_2d = np.array([[x[2],x[3]] for x in iris.data])

# Establish a tensorflow graph session
sess = tf.Session()

# Define batch size
batch_size = 20


# Establish tensorflow 2D placeholders for graph session data feeding of
# iris features and classification targets
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],  dtype=tf.float32)

# Define the variable A weight and b bias for the tensorflow graph session 
# to learn and modify

# Initialize weight A with a random value 
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize bias b with a random value 
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Define the graph model operations
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult,b)
my_output = tf.subtract(x1_data, my_add)

# Define the loss function over each batch
xloss =\
tf.nn.sigmoid_cross_entropy_with_logits(labels = y_target, logits = my_output)

# Initialize variable for graphic session
init = tf.global_variables_initializer()
sess.run(init)

# Define gradient descent optimization algorithm for graphic session
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = my_opt.minimize(xloss)

# Define number of sessions for training
sess_num = 1000
# loop for batch graphic training sessions
for i in range(0,sess_num):
    # Generate random number based on the size of the training dataset
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    # Select random x feature value and y target value for training
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    # run graphic training session with feed placeholder data
    sess.run(train_step,\
             feed_dict={x1_data:rand_x1, x2_data:rand_x2, y_target:rand_y})
    # Display algorithm status after each block on 200 iterations
    if ((i+1)%200 == 0):
        print('Step#{} A = {} b = {}'.format((i+1),sess.run(A),sess.run(b)))
        # temp_loss = tf.reduce_mean(sess.run(xloss,\
              #feed_dict={x1_data:rand_x1, x2_data:rand_x2, y_target:rand_y}))
        # print('Loss = {}'.format(temp_loss))

#Visualize petal features with decision boundary
        
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append((slope*i)+intercept)
    
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]

plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize = 20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()