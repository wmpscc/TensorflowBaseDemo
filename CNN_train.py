#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 25


def weights_with_loss(shape, stddev, wl):
    var = tf.truncated_normal(stddev=stddev, shape=shape)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return tf.Variable(var)


def biasses(shape):
    i = tf.constant(0.1, shape=shape)
    return tf.Variable(i)


def conv(image, filter):
    return tf.nn.conv2d(image, filter=filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(image):
    return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # ksize 池化核大小


def net(image, drop_pro):
    W_conv1 = weights_with_loss([5, 5, 3, 32], 5e-2, wl=0.0)
    b_conv1 = biasses([32])
    conv1 = tf.nn.relu(conv(image, W_conv1) + b_conv1)
    pool1 = max_pool_2x2(conv1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1, alpha=0.001 / 9.0, beta=0.75)

    W_conv2 = weights_with_loss([5, 5, 32, 64], stddev=5e-2, wl=0.0)
    b_conv2 = biasses([64])
    conv2 = tf.nn.relu(conv(norm1, W_conv2) + b_conv2)
    norm2 = tf.nn.lrn(conv2, 4, bias=1, alpha=0.001 / 9.0, beta=0.75)
    pool2 = max_pool_2x2(norm2)


    W_conv3 = weights_with_loss([5, 5, 64, 128], stddev=0.04, wl=0.004)
    b_conv3 = biasses([128])
    conv3 = tf.nn.relu(conv(pool2, W_conv3) + b_conv3)
    pool3 = max_pool_2x2(conv3)

    W_conv4 = weights_with_loss([5, 5, 128, 256], stddev=1/128, wl=0.004)
    b_conv4 = biasses([256])
    conv4 = tf.nn.relu(conv(pool3, W_conv4) + b_conv4)
    pool4 = max_pool_2x2(conv4)

    image_raw = tf.reshape(pool4, shape=[-1, 8 * 8 * 256])

    # 全连接层
    fc_w1 = weights_with_loss(shape=[8 * 8 * 256, 1024], stddev=1/256, wl=0.0)
    fc_b1 = biasses(shape=[1024])
    fc_1 = tf.nn.relu(tf.matmul(image_raw, fc_w1) + fc_b1)

    # drop-out层
    drop_out = tf.nn.dropout(fc_1, drop_pro)

    fc_2 = weights_with_loss([1024, 10], stddev=0.01, wl=0.0)
    fc_b2 = biasses([10])

    return tf.matmul(drop_out, fc_2) + fc_b2



def get_accuracy(logits, label):
    current = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), 'float')
    accuracy = tf.reduce_mean(current)
    return accuracy


# 读训练集数据
def read_train_data():
    reader = tf.TFRecordReader()
    filename_train = tf.train.string_input_producer(["TFRecord128/train.tfrecords"])
    _, serialized_example_test = reader.read(filename_train)
    features = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    img_train = features['image_raw']
    images_train = tf.decode_raw(img_train, tf.uint8)
    images_train = tf.reshape(images_train, [128, 128, 3])
    labels_train = tf.cast(features['label'], tf.int64)
    labels_train = tf.cast(labels_train, tf.int64)
    labels_train = tf.one_hot(labels_train, 10)
    return images_train, labels_train


# 读测试集数据
def read_test_data():
    reader = tf.TFRecordReader()
    filename_test = tf.train.string_input_producer(["TFRecord128/test.tfrecords"])
    _, serialized_example_test = reader.read(filename_test)
    features_test = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img_test = features_test['image_raw']
    images_test = tf.decode_raw(img_test, tf.uint8)
    images_test = tf.reshape(images_test, [128, 128, 3])
    labels_test = tf.cast(features_test['label'], tf.int64)
    labels_test = tf.one_hot(labels_test, 10)
    return images_test, labels_test

def save_model(sess, step):
    MODEL_SAVE_PATH = "./model128/"
    MODEL_NAME = "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)

def train():
    x_train, y_train = read_train_data()
    x_test, y_test = read_test_data()
    x_batch_train, y_batch_train = tf.train.shuffle_batch([x_train, y_train], batch_size=BATCH_SIZE, capacity=200,
                                                          min_after_dequeue=100, num_threads=3)
    x_batch_test, y_batch_test = tf.train.shuffle_batch([x_test, y_test], batch_size=BATCH_SIZE, capacity=200,
                                                        min_after_dequeue=100, num_threads=3)

    x = tf.placeholder(tf.float32, shape=[None, 49152])
    y = tf.placeholder(tf.int64, shape=[None, 10])
    drop_pro = tf.placeholder('float')

    images = tf.reshape(x, shape=[BATCH_SIZE, 128, 128, 3])

    logits = net(images, drop_pro)

    getAccuracy = get_accuracy(logits, y)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    global_step = tf.Variable(0, name='global_step')
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy, global_step=global_step)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(2500):
        images_train, label_train = sess.run([x_batch_train, y_batch_train])
        _images_train = np.reshape(images_train, [BATCH_SIZE, 49152])
        if i % 100 == 0:
            accuracy = sess.run(getAccuracy, feed_dict={x: _images_train, y: label_train, drop_pro: 1})
            loss = sess.run(cross_entropy, feed_dict={x: _images_train, y: label_train, drop_pro: 1})
            print("step(s): %d ----- accuracy: %g -----loss: %g" % (i, accuracy, loss))
        sess.run(train_op, feed_dict={x: _images_train, y: label_train, drop_pro: 0.5})

    images_test, label_test = sess.run([x_batch_test, y_batch_test])
    _images_test = np.reshape(images_test, [BATCH_SIZE, 49152])
    accuracy_test = sess.run(getAccuracy, feed_dict={x: _images_test, y: label_test, drop_pro: 1})
    print("test accuracy: %g" % accuracy_test)

    save_model(sess, i)
    coord.request_stop()
    coord.join(threads)


train()
