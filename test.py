from tensorflow import keras
import cv2
import os
import numpy as np

generator_a_to_b = keras.models.load_model('models/generator_a_to_b.h5')
generator_b_to_a = keras.models.load_model('models/generator_b_to_a.h5')

discriminator_a = keras.models.load_model('models/discriminator_a.h5')
discriminator_b = keras.models.load_model('models/discriminator_b.h5')

for image_path in os.listdir('images_a/images/'):
    image = cv2.imread('images_a/images/' + image_path)
    image = cv2.resize(image, 256, 256)
    image_translated = generator_a_to_b.predict(image)
    image_confidence = discriminator_b.predict(image_translated)
    cv2.imsave('output_' + image_path + '_translated_' + str(image_confidence))

for image_path in os.listdir('images_b/images/'):
    image = cv2.imread('images_b/images/' + image_path)
    image = cv2.resize(image, 256, 256)
    image_translated = generator_b_to_a.predict(image)
    image_confidence = discriminator_a.predict(image_translated)
    cv2.imsave('output_' + image_path + '_translated_' + str(image_confidence))
