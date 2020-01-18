from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from skimage.io import imread

from efficientnet.keras import EfficientNetB0, center_crop_and_resize, preprocess_input


def build():
    # Downloads label mappings on first run
    decode_predictions(np.empty([1, 1000]))
    # Downloads weights on first run
    setup()

def setup():
    return EfficientNetB0(weights='imagenet')

def infer(model, image_path):
    image = imread(image_path)
    image_size = model.input_shape[1]
    x = center_crop_and_resize(image, image_size=image_size)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)
    predictions = model.predict(x)
    predictions_with_labels = decode_predictions(predictions)
    return {t[1]: t[2] for t in predictions_with_labels[0]}
