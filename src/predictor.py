import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image_dataset import load_image
from tensorflow.python.ops import image_ops


def get_processed_image_from_url(image_url, image_size, num_channels, interpolation):
    """
    Load an image from a url and resize it.
    Adapted from tensorflow.python.keras.preprocessing.image_dataset.load_image
    """
    img = requests.get(image_url).content
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


IMG_SIZE = (160, 160)

class Predictor(object):
    model = None

    @classmethod
    def get_model(cls):
        """Load the model object if it's not already loaded."""
        if cls.model is None:
            cls.model = load_model('models/model')
        return cls.model

    @classmethod
    def get_normalized_prediction(cls, prediction):
        """Return prediction as sigmoid-normalized float."""
        return 1.0 / (1.0 + np.exp(-1.0 * prediction[0][0]))

    @classmethod
    def predict_from_image_path(cls, image_path):
        loaded_model = cls.get_model()
        loaded_image = load_image(image_path, image_size=IMG_SIZE, interpolation='bilinear', num_channels=3)
        loaded_image = np.expand_dims(loaded_image, axis=0)
        raw_prediction = loaded_model.predict(loaded_image)
        return cls.get_normalized_prediction(raw_prediction)

        # Normalize with sigmoid function for more intuitive 0.0-1.0 output
        normalized_prediction = 1.0 / (1.0 + np.exp(-1.0 * raw_prediction))
        return normalized_prediction

    @classmethod
    def predict_from_image_url(cls, image_url):
        loaded_model = cls.get_model()
        loaded_image = get_processed_image_from_url(image_url, image_size=IMG_SIZE, interpolation='bilinear', num_channels=3)
        loaded_image = np.expand_dims(loaded_image, axis=0)
        raw_prediction = loaded_model.predict(loaded_image)
        return cls.get_normalized_prediction(raw_prediction)
