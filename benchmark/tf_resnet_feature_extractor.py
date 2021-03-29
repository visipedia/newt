import os

import numpy as np
import tensorflow as tf
import tqdm

import configs
from tf_resnet import resnet50

class TFResNet50FeatureExtractor():

    def __init__(self, input_height=224, input_width=224, weights=None, width_multiplier=1, center_crop=True, device=None):

        self.input_height = input_height
        self.input_width = input_width
        self.weights = weights
        self.width_multiplier = width_multiplier
        self.center_crop = center_crop
        self.device = device

        assert os.path.exists(self.weights)

        self.build_model()

    def build_model(self):

        backbone = resnet50(
            include_top=False,
            weights=self.weights,
            input_shape=(self.input_height, self.input_width, 3),
            width_multiplier=self.width_multiplier
        )

        self.model = backbone

    def preprocess(self, image_fp):

        contents = tf.io.read_file(image_fp)
        image = tf.io.decode_image(contents, channels=3, dtype=tf.dtypes.uint8, expand_animations=False)

        # Convert the image to [0, 1]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.center_crop:
            image = tf.keras.preprocessing.image.smart_resize(image, [self.input_height, self.input_width], interpolation='bilinear')
        else:
            image = tf.image.resize(image, [self.input_height, self.input_width], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        image = tf.reshape(image, [self.input_height, self.input_width, 3])
        image = tf.clip_by_value(image, 0., 1.)

        return image

    def extract_features(self, image_fp):
        """ Load the image and extract features.
        """

        processed_image = self.preprocess(image_fp)
        features = self.model(tf.expand_dims(processed_image, 0), training=False)

        return features[0].numpy()

    def extract_features_batch(self, image_fp_list, batch_size=32, use_pbar=False):

        with tf.device("CPU:0"):
            dataset = tf.data.Dataset.from_tensor_slices(image_fp_list)
            dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size, drop_remainder=False)

        dataset_iter = iter(dataset)

        features = np.empty([len(image_fp_list), self.model.output.shape[1]])
        feature_index = 0

        if use_pbar:
            dataset_iter = tqdm.tqdm(dataset_iter, leave=False)

        with tf.device(self.device):

            for batch in dataset_iter:

                batch_features = self.model(batch, training=False)
                num_in_batch = batch.shape[0]
                features[feature_index:feature_index+num_in_batch] = batch_features.numpy()
                feature_index += num_in_batch

        return features

def load_feature_extractor(model_spec, device=None, input_height=224, input_width=224, center_crop=True):
    """ The tensorflow resnet50 models have all been standardized. So we simply load the specified weights into the resnet50 model.
    """

    assert model_spec['backbone'] in [configs.RESNET50, configs.RESNET50_X4], "Unsupported tensorflow feature extractor: %s" % model_spec['backbone']

    model_weights_fp = model_spec['weights']

    width_multiplier = 1
    if model_spec['backbone'] == configs.RESNET50_X4:
        width_multiplier = 4

    if device is not None:
        with tf.device(device):
            feature_extractor = TFResNet50FeatureExtractor(
                input_height=input_height,
                input_width=input_width,
                weights=model_weights_fp,
                width_multiplier=width_multiplier,
                center_crop=center_crop,
                device=device
            )
    else:
        feature_extractor = TFResNet50FeatureExtractor(
            input_height=input_height,
            input_width=input_width,
            weights=model_weights_fp,
            width_multiplier=width_multiplier,
            center_crop=center_crop,
        )

    return feature_extractor