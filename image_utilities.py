import numpy as np
import os
import PIL.Image
import tensorflow as tf

from shutil import copyfile


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def load_img(path_to_img, target_type, max_dim=None, copy=True):

    if copy:
        extension = path_to_img.split(".")[-1]
        # Try to download from url
        try:
            path_to_img = tf.keras.utils.get_file(os.path.join(os.getcwd(), 'images', '%s.%s' % (target_type, extension)), path_to_img)
        # Else assume it is local
        except ValueError:
            new_path = os.path.join(os.getcwd(), 'images', '%s.%s' % (target_type, extension))
            copyfile(path_to_img, new_path)
            path_to_img = new_path

    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)