import tensorflow as tf
import image_utils
import os
import warnings

tf.logging.set_verbosity(tf.logging.INFO)


class CNNClassifier():

    def __init__(self):

        # Set CNN model default parameters
        self.dataset_dir = 'dataset'
        self.data_dir = self.dataset_dir + '\\output'
        self.labels_file = self.dataset_dir + '\\labels.txt'
        self.train_list, self.validation_list = image_utils.get_files_list(self.data_dir)
        self.features_shape = 250
        self.num_channels = 3
        self.batch_size = 32
        self.num_epochs = 1

        self._set_num_classes()
        self._set_num_images()

        # Set image utils params
        image_utils.params["feature_shape"] = self.features_shape
        image_utils.params["num_channels"] = self.num_channels

        # Define flags and log
        tf.logging.set_verbosity(tf.logging.INFO)
        self._define_flags()

        # Build classifier
        self._classifier_builder()

        # Dump to system out the characteristics of the training / validation sets
        self._dump()

    def _set_num_classes(self):
        self.num_classes = 0

        with open(self.labels_file) as f:
            for _ in f:
                self.num_classes = self.num_classes + 1
            f.close()

    def _set_num_images(self):
        train_dir = self.dataset_dir + '\\train'
        validation_dir = self.dataset_dir + '\\validation'

        num_images_train = 0
        num_images_validation = 0

        # Count images in train set
        for _, _, files in os.walk(train_dir):
            for file in files:
                num_images_train = num_images_train + 1

        # Count images in validation set
        for _, _, files in os.walk(validation_dir):
            for _ in files:
                num_images_validation = num_images_validation + 1

        self.num_images = {
            'train': num_images_train,
            'validation': num_images_validation
        }


    def _define_flags(self):
        ''''''
        tf.flags.DEFINE_string('output_directory', self.data_dir, 'Output data directory')
        self.flags = tf.flags.FLAGS

    def _dump(self):

        print('')
        print('---------------------------------------------------------------')
        print('Building classifier for ' + str(self.num_classes) + ' classes')
        print('Found ' + str(self.num_images['train']) + ' images in train set')
        print('Found ' + str(self.num_images['validation']) + ' images in validation set')
        print('---------------------------------------------------------------')
        print('')

    def cnn_model_fn(self, features, labels, mode):
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
        input_layer = tf.reshape(features['image'], [-1, self.features_shape, self.features_shape, 3])

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
        pool2_shape = pool2.get_shape()
        pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])

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
            units=self.num_classes
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
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                                   depth=self.num_classes)  # must use onehot encoded labels
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

    def dataset_input_fn(self, is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
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

        image_utils.params["is_trainning"] = is_training

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

    def train_input_fn(self, filenames):
        return self.dataset_input_fn(
            is_training=True,
            filenames=filenames,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs
        )

    def _classifier_builder(self):
        self.classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn,
            model_dir=os.path.join(self.flags.output_directory, "tb")
        )

    def _set_logging(self):
        tensors_to_log = {"probabilities": "softmax_tensor"}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    def train(self, steps=10, set_logging=True):
        '''
        Train the classifier

        :param steps: Number of stepts which to train the model
        :param set_logging: Boolean specifying if you want to set logging hooks or not
        :return:
        '''
        if set_logging:
            self._set_logging()

        self.classifier.train(input_fn=lambda: self.train_input_fn(self.train_list), steps=steps,
                              hooks=[self.logging_hook])


def main(unused_arg):
    print('')
    model = CNNClassifier()
    model.train(steps=100)


if __name__ == '__main__':
    tf.app.run()
