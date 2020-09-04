import tensorflow as tf
import sys
# Converting a SavedModel to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_saved_model("./saved_model/my_model")
tflite_model = converter.convert()


with open('MNIST_model.tflite', 'wb') as w:
    w.write(tflite_model)