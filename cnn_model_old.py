from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import image_utils

tf.logging.set_verbosity(tf.logging.INFO)

FEATURES_SHAPE = 0
NUM_CATEGORIES = 0
NUM_CHANNELS = 0
BATCH_SIZE = 0
NUM_EPOCHS = 0
NUM_IMAGES = {}


def cnn_model_fn(features, labels, mode):
    '''
    Build a ConvNet as described in A Guide to TF Layers: Building a Convolutional Neural Network
    https://www.tensorflow.org/tutorials/layers

    :param features: Images
    :param labels: Labels
    :param mode: PREDICTION, TRAIN or EVAL
    :param features_shape: shape of your images (image must be feature_shape x feature_shape with 3 channels RGB)
    :param num_categories: number of categories being predicted (number of labels)
    :return:
    '''
    # Input layer
    input_layer = tf.reshape(features['image'], [-1, FEATURES_SHAPE, FEATURES_SHAPE, 3])

    # 1st Convolutional and Pooling Layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # 2nd Convolutional and Pooling Layers
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # Flattening
    pool2_flat = tf.reshape(pool2, [-1, 62 * 62 * 64])

    # Dense layer
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )

    # Dropout regularization method in layers to impprove the results
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits layer: probabilities mapped to [-inf,+inf] to input in softmax
    logits = tf.layers.dense(
        inputs=dropout,
        units=NUM_CATEGORIES
    )

    # Generate predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # PREDICTION MODE ------------------------------------------------

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    # TRAINING MODE --------------------------------------------------

    # Performing the softmax activation on logits, calculating cross entropy and return loss as scalar Tensor
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    print(onehot_labels)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Training: configure model to optimize the loss by performing the stochastic gradient descent optimization
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op
        )

    # EVALUATION MODE -----------------------------------------------
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def dataset_input_fn(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
    '''
    Build a input_fn of a tf.estimator.Estimator
    Based on tensorflow documentation: https://www.tensorflow.org/programmers_guide/datasets

    :param is_training:
    :param filenames:
    :param batch_size:
    :param num_epochs:
    :return:
    '''

    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    # Convert each element of dataset to TFRecord
    image_utils.NUM_CHANNELS = NUM_CHANNELS
    image_utils.IS_TRAINNING = is_training

    dataset = dataset.map(lambda value: image_utils.record_parser(value, is_training),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    return features, labels


def train_input_fn(filenames):
    return dataset_input_fn(
        is_training=True,
        filenames=filenames,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )


def validation_input_fn(filenames):
    return dataset_input_fn(
        is_training=False,
        filenames=filenames,
        batch_size=BATCH_SIZE,
        num_epochs=1
    )
