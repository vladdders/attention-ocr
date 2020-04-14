from __future__ import absolute_import

import math
import sys

import numpy as np
import tensorflow as tf

from PIL import Image
from six import BytesIO as IO

from .bucketdata import BucketData

TFRecordDataset = tf.data.TFRecordDataset  # pylint: disable=invalid-name


class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32
    CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    @staticmethod
    def set_full_ascii_charmap():
        DataGen.CHARMAP = ['', '', ''] + [chr(i) for i in range(32, 127)]

    def __init__(self,
                 annotation_fn,
                 buckets,
                 epochs=1000,
                 max_width=None,
                 max_height=None,
                 channels=1):
        """
        :param annotation_fn:
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.channels = channels
        self.epochs = epochs

        self.original_max_width = max_width
        self.original_max_height = max_height

        max_resized_width = 1. * max_width / max_height * DataGen.IMAGE_HEIGHT
        self.max_width = int(math.ceil(max_resized_width))

        self.height = DataGen.IMAGE_HEIGHT

        self.bucket_specs = buckets
        self.bucket_data = BucketData()

        dataset = TFRecordDataset([annotation_fn])
        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        self.bucket_data = BucketData()

    def gen(self, batch_size):

        dataset = self.dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        images, labels, comments = iterator.get_next()
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:

            while True:
                try:
                    raw_images, raw_labels, raw_comments = sess.run([images, labels, comments])
                    for img, lex, comment in zip(raw_images, raw_labels, raw_comments):

                        if self.original_max_width and (Image.open(IO(img)).size[0] <= self.original_max_width):
                            word = self.convert_lex(lex)

                            bucket_size = self.bucket_data.append(self._prepare_image_custom(img), word, lex, comment)
                            if bucket_size >= batch_size:
                                bucket = self.bucket_data.flush_out(
                                    self.bucket_specs,
                                    go_shift=1)
                                yield bucket

                except tf.errors.OutOfRangeError:
                    break

        self.clear()

    def convert_lex(self, lex):
        if sys.version_info >= (3,):
            lex = lex.decode('iso-8859-1')

        assert len(lex) < self.bucket_specs[-1][1]

        return np.array(
            [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
            dtype=np.int32)

    @staticmethod
    def _parse_record(example_proto):
        features = tf.io.parse_single_example(
            serialized=example_proto,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
                'comment': tf.io.FixedLenFeature([], tf.string, default_value=''),
            })
        return features['image'], features['label'], features['comment']

    def _prepare_image(self, image):
        """Resize the image to a maximum height of `self.height` and maximum
        width of `self.width` while maintaining the aspect ratio. Pad the
        resized image to a fixed size of ``[self.height, self.width]``.

        This method used to be within the graph. I moved it here and it's not
        used but it might be useful in the future.

        """
        img = tf.image.decode_jpeg(image, channels=self.channels)
        dims = tf.shape(input=img)
        dims = tf.cast(x=dims, dtype=tf.float32)

        width = self.max_width
        max_width = tf.cast(tf.math.ceil(tf.truediv(dims[1], dims[0]) * DataGen.IMAGE_HEIGHT), dtype=tf.int32)

        max_height = tf.cast(tf.math.ceil(tf.truediv(tf.cast(x=width, dtype=tf.float32),
                                                    tf.cast(x=max_width, dtype=tf.float32)) * self.height_float),
                             dtype=tf.int32)

        resized = tf.cond(
            # if width is greater or equal than max_width
            # it returns the output of the satisfied condition below
            # else it returns the image resized to max_height and width
            pred=tf.greater_equal(width, max_width),
            true_fn=lambda: tf.cond(
                # if width is less or equal than max width
                # it returns the image
                # else it returns the resized image with dims self.height and max width
                pred=tf.less_equal(tf.cast(x=dims[0], dtype=tf.int32), self.height),
                true_fn=lambda: tf.cast(img, dtype=tf.float32),
                false_fn=lambda: tf.image.resize(img, [self.height, max_width],
                                               method=tf.image.ResizeMethod.BICUBIC),
            ),
            false_fn=lambda: tf.image.resize(img, [max_height, width],
                                           method=tf.image.ResizeMethod.BICUBIC)
        )

        padded = tf.image.pad_to_bounding_box(resized, 0, 0, self.height, width).eval()

        return padded

    def _prepare_image_custom(self, image):
        """
        This function pre-processes the image before training. Make sure you apply the same
        operations when carrying out the inference. In the original model the pre-processing
        part took part within the graph, but some of those operations are not supported
        in tflite thus it's been moved outside the graph. Currently, the model supports
        only images of height 32 because of the way the convolutional part handles the
        tensor shapes.
        @TODO: adapt model to variable height images
        Args:
            image:
        Returns:
        """
        img = Image.open(IO(image))
        img = img.resize((self.max_width, self.height))
        img = np.array(img)
        img = np.expand_dims(img, 2)

        return img