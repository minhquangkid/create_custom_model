import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
print(tf.__version__) #version 2.9.3


assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)




# train_data = object_detector.DataLoader.from_pascal_voc(
#     'android_figurine/train',
#     'android_figurine/train',
#     ['android', 'pig_android']
# )

# val_data = object_detector.DataLoader.from_pascal_voc(
#     'android_figurine/validate',
#     'android_figurine/validate',
#     ['android', 'pig_android']
# )



train_data = object_detector.DataLoader.from_pascal_voc(
    'android_figurine/train_fishing_float',
    'android_figurine/train_fishing_float',
    ['fishing float']
)

val_data = object_detector.DataLoader.from_pascal_voc(
     'android_figurine/validation_fishing_float',
     'android_figurine/validation_fishing_float',
     ['fishing float']
)

spec = model_spec.get('efficientdet_lite0')


model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=10, validation_data=val_data) #the higher epochs, the exactly it will be

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='fishing_float_model.tflite')

# model.evaluate_tflite('android.tflite', val_data)

# # Download the TFLite model to your local computer.
# from google.colab import files
# files.download('android.tflite')