from tensorflow import keras
import cv2
import os
import numpy as np
import utility_functions as uf

generator_a_to_b = keras.models.load_model('models/generator_a_to_b.h5')
generator_b_to_a = keras.models.load_model('models/generator_b_to_a.h5')

discriminator_a = keras.models.load_model('models/discriminator_a.h5')
discriminator_b = keras.models.load_model('models/discriminator_b.h5')

uf.save_outputs(generator_a_to_b, discriminator_b, 'images_a/images/')
uf.save_outputs(generator_b_to_a, discriminator_a, 'images_b/images/')
