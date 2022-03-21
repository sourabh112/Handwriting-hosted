#File to convert H5 model to tflite model
import tensorflow as tf
from keras.models import model_from_json
new_model = model_from_json(open('trying.json').read())
new_model.load_weights('model.hdf5')
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)