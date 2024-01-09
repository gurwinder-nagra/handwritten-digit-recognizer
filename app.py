import os
os.system("pip uninstall -y gradio")
os.system("pip install gradio==3.50.2")
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import gradio as gr

# Load the model
model = tf.keras.models.load_model('handwritten_digits.model')

def recognize_digit(image):
  if image is not None:
    image = image.reshape((1,28,28,1)).astype('float32')/255
    prediction = model.predict(image)
    return {str(i) : float(prediction[0][i]) for i in range(10)}
  else:
    return ''

iface = gr.Interface(
    fn = recognize_digit,
    inputs=gr.Image(shape=(28,28),image_mode = 'L',invert_colors=True, source = 'canvas'),
    outputs=gr.Label(top_num_classes=3))

iface.launch()