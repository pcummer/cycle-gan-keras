import cv2
import os
import numpy as np

def save_outputs(generator, discriminator, directory):
    for image_path in os.listdir(directory):
        image = cv2.imread(directory + image_path)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis = 0)
        image_translated = generator.predict(image)
        image_confidence = discriminator.predict(image_translated)
        image_translated = np.squeeze(image_translated)
        cv2.imwrite('output/' + image_path + '_translated_' + str(image_confidence) + '.png', image_translated)
