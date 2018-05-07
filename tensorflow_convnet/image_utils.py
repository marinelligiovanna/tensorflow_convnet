import glob
import tensorflow as tf
import vgg_preprocessing
import os
from PIL import Image

params = {
    "num_channels": 0,
    "is_trainning": False,
    "feature_shape": 0
}


def get_files_list(data_dir):
    train_list = glob.glob(data_dir + '/' + 'train-*')
    validation_list = glob.glob(data_dir + '/' + 'validation-*')

    if len(train_list) == 0:
        raise IOError('No files found at train set file path')

    if len(validation_list) == 0:
        raise IOError('No files found at validation set file path')

    return train_list, validation_list


def record_parser(raw_record, is_training):
    '''

    Use `tf.parse_single_example()` to extract data from a `tf.Example`
    protocol buffer, and perform any additional per-record preprocessing.
    Based on tensorflow documentation and Brijesh Thumar tutorial 'Feed
    your own data set into the CNN model in Tensorflow'

    https://www.tensorflow.org/programmers_guide/datasets
    https://github.com/Thumar/cnn_dog_vs_cat/blob/master/cnn_dog_cat.py

    :param raw_record:
    :param is_training:
    :return:
    '''

    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]),
        params["num_channels"])

    # Note that tf.image.convert_image_dtype scales the image data to [0, 1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=params["feature_shape"],
        output_width=params["feature_shape"],
        is_training=is_training)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32)

    return {"image": image}, label


def save_jpeg_tf_format(old_data_folder, new_data_folder):
    imagePaths = glob.glob(old_data_folder + '\\' + '*.jpg')

    for imagePath in imagePaths:
        print(imagePath)
        image = Image.open(imagePath)
        image.save(new_data_folder + '\\' + os.path.basename(imagePath))
