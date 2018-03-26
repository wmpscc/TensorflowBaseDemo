#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 500
IMAGE_SIZE = 128

def weights(shape):
    i = tf.truncated_normal(stddev=0.3, shape=shape)
    return tf.Variable(i)


def biasses(shape):
    i = tf.constant(0.1, shape=shape)
    return tf.Variable(i)


def conv(image, filter):
    return tf.nn.conv2d(image, filter=filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(image):
    return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # ksize 池化核大小


def net(image, drop_pro):
    W_conv1 = weights([5, 5, 3, 32])
    b_conv1 = biasses([32])
    conv1 = tf.nn.relu(conv(image, W_conv1) + b_conv1)
    pool1 = max_pool_2x2(conv1)

    W_conv2 = weights([5, 5, 32, 64])
    b_conv2 = biasses([64])
    conv2 = tf.nn.relu(conv(pool1, W_conv2) + b_conv2)
    pool2 = max_pool_2x2(conv2)

    W_conv3 = weights([5, 5, 64, 128])
    b_conv3 = biasses([128])
    conv3 = tf.nn.relu(conv(pool2, W_conv3) + b_conv3)
    pool3 = max_pool_2x2(conv3)

    W_conv4 = weights([5, 5, 128, 256])
    b_conv4 = biasses([256])
    conv4 = tf.nn.relu(conv(pool3, W_conv4) + b_conv4)
    pool4 = max_pool_2x2(conv4)
    image_raw = tf.reshape(pool4, shape=[-1, int((IMAGE_SIZE / 16) * (IMAGE_SIZE / 16) * 256)])

    # 全连接层
    fc_w1 = weights(shape=[int((IMAGE_SIZE / 16) * (IMAGE_SIZE / 16) * 256), 1024])
    fc_b1 = biasses(shape=[1024])
    fc_1 = tf.nn.relu(tf.matmul(image_raw, fc_w1) + fc_b1)

    # drop-out层
    drop_out = tf.nn.dropout(fc_1, drop_pro)

    fc_2 = weights([1024, 10])
    fc_b2 = biasses([10])

    return tf.matmul(drop_out, fc_2) + fc_b2


def get_accuracy(logits, label):
    current = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), 'float')
    accuracy = tf.reduce_mean(current)
    return accuracy


# 读测试集数据
def read_test_data():
    reader = tf.TFRecordReader()
    if IMAGE_SIZE == 128:
        filename_test = tf.train.string_input_producer(["TFRecord128/test.tfrecords"])
    else:
        filename_test = tf.train.string_input_producer(["TFRecord64/test.tfrecords"])
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
    images_test = tf.reshape(images_test, [IMAGE_SIZE, IMAGE_SIZE, 3])
    labels_test = tf.cast(features_test['label'], tf.int64)
    labels_test = tf.one_hot(labels_test, 10)
    return images_test, labels_test


def train():
    x_test, y_test = read_test_data()
    x_batch_test, y_batch_test = tf.train.shuffle_batch([x_test, y_test], batch_size=BATCH_SIZE, capacity=2000,
                                                        min_after_dequeue=100, num_threads=3)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2*3])
    y = tf.placeholder(tf.int64, shape=[None, 10])
    drop_pro = tf.placeholder('float')
    images = tf.reshape(x, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    logits = net(images, drop_pro)
    getAccuracy = get_accuracy(logits, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if IMAGE_SIZE == 128:
            MODEL_PATH = "model128/"
        else:
            MODEL_PATH = "model64/"
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            #加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过文件名得到模型保存是迭代的轮数
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            images_test, label_test = sess.run([x_batch_test, y_batch_test])
            _images_test = np.reshape(images_test, [BATCH_SIZE, IMAGE_SIZE**2*3])
            accuracy_test = sess.run(getAccuracy, feed_dict={x: _images_test, y: label_test, drop_pro: 1})
            print("Image size: %s -- After %s training step(s), validation accuracy = %g" % (IMAGE_SIZE, global_step, accuracy_test))
        else:
            print("error!")

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()