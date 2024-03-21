import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
print(tf.__version__) #version 2.9.0

# Set the global mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)




train_data = object_detector.DataLoader.from_pascal_voc(
    'android_figurine/train',
    'android_figurine/train',
    ['android', 'pig_android']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'android_figurine/validate',
    'android_figurine/validate',
    ['android', 'pig_android']
)

spec = model_spec.get('efficientdet_lite0')


model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='android.tflite')

# model.evaluate_tflite('android.tflite', val_data)

# # Download the TFLite model to your local computer.
# from google.colab import files
# files.download('android.tflite')