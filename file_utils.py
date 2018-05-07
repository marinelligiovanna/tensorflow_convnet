import glob
import vgg_preprocessing
import tensorflow as tf

def get_dataset_list(data_dir):
    train_list = glob.glob(data_dir + '/' + 'train-*')
    validation_list = glob.glob(data_dir + '/' + 'validation-*')

    if len(train_list) == 0:
        raise IOError('No train set found')

    return train_list, validation_list

def parse_record(record, is_training, num_channels, num_classes, img_shape):

    '''
    Based on: http://gitlab.heyuantao.cn/heyuantao/object_detection/blob/4d053deb0cae63f748928ba8d18802d4e0be2692/official/resnet/imagenet_main.py

    :param record:
    :param is_training:
    :param num_channels:
    :param num_classes:
    :param img_shape:
    :return:
    '''

    features = {
        'image/encoded':
            tf.FixedLenFeature((), dtype=tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), dtype=tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=''),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }

    parsed = tf.parse_single_example(
        record,
        features
    )

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'],
                   shape=[]
        ),
        num_channels
    )

    image = tf.image.convert_image_dtype(
        image,
        dtype=tf.float32
    )

    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=img_shape,
        output_width=img_shape,
        is_training=is_training
    )

    label = tf.cast(
        tf.reshape(parsed['image/class/label'],
                   shape=[]
        ),
        dtype=tf.int32
    )

    return image, tf.one_hot(label, num_classes)
