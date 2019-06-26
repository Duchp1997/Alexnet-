# _*_ coding :utf-8 _*_
# author:du changping
# date: 2019/6/22 20:20
# file_name: main.py
# develop_tool: PyCharm
# 不要砸电脑

from datetime import datetime
import math
import tensorflow as tf
import time


def print_activation(t):
    """
    本函数用来输出神经网络的参数介绍
    :param t:
    :return:
    """
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    """
    Alexnet 网络主体结构
    :param images:
    :return:
    """
    parameters = []

    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],  #  这里[11, 11, 3, 64]表示产生11*11，通道为3，核数量为64
                             dtype=tf.float32,
                             stddev=1e-1), name='weights')

        conv = tf.nn.conv2d(input=images, filter=kernel, strides=[1, 4, 4, 1], padding="SAME",  )
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=64), trainable=True, name="biases")
        conv1 = tf.nn.bias_add(conv, biases)
        print_activation(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(input=conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75,name="lrn1")
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2,1], padding="VALID", name="pool1")
    print_activation(pool1)

    with tf.variable_scope(conv2) as scope:
        kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 64, 192],
                                                               dtype=tf.float32,
                                                               stddev=1e-1),
                             name="weights")
        conv = tf.nn.conv2d(input=pool1, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters +=[kernel, biases]
        print_activation(conv2)

    lrn2 = tf.nn.lrn(input=conv2, 4, bias = 1.0, alpha=0.001/9, beta=0.75, name="lrn2")
    pool2 = tf.nn.max_pool(input=lrn2, ksize=[1, 3, 3, 1],
                           padding="VALID", strides=[1, 2, 2, 1],
                           name="pool2" )
    print_activation(pool2)

    with tf.variable_scope("conv3") as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 384]), dtype=tf.float32, name="weights")
        conv = tf.nn.conv2d(input=pool2, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384],
                             dtype=tf.float32), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name="scope")
        parameters += [kernel, biases]
        print_activation(conv3)

    with tf.variable_scope("conv4") as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 256]), dtype=tf.float32, name="weights")
        conv = tf.nn.conv2d(input=conv3, filter=kernel, strides=[1,1,1,1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable= True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv4)

    with tf.variable_scope("conv5") as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[3,3,256,256]), dtype=tf.float32, name="weights")
        conv = tf.nn.conv2d(input=conv4, filter=kernel, strides=[1,1,1,1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256],dtype=tf.float32), name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name="scope")
        parameters += [kernel, biases]
        print_activation(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1],
                           padding="VALID", name="pool5")
    print_activation(pool5)
    # TODO 这里关于全连接层的编程内容有待思考

    return pool5, parameters









