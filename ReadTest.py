#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, labels):
    for i in np.arange(0, 20):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(labels[i], fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

reader = tf.TFRecordReader()
filename_train = tf.train.string_input_producer(["TFRecord/train.tfrecords"])
_, serialized_example_test = reader.read(filename_train)
features = tf.parse_single_example(
    serialized_example_test,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    }
)
img = features['image_raw']
images_train = tf.decode_raw(features['image_raw'], tf.uint8)
images_train = tf.reshape(images_train, [128, 128, 3])
labels_train = tf.cast(features['label'], tf.int64)

x_batch_train, y_batch_train = tf.train.shuffle_batch([images_train, labels_train], batch_size=25, capacity=200,
                                                      min_after_dequeue=100, num_threads=3)
labels_train = tf.reshape(y_batch_train, [25])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image, label = sess.run([x_batch_train, labels_train])
    plot_images(image, label)
    coord.request_stop()
    coord.join(threads)
