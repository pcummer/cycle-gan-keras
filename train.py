import model_zoo
import cv2
import numpy as np
from tensorflow import keras

number_training_batches = 50
training_batch_size = 128

# create the models
gan_a_to_b, gan_b_to_a, generator_a_to_b, generator_b_to_a, discriminator_a, discriminator_b = model_zoo.create_gans(nodes_per_layer=32)

# simple data augmentation implicitly including resizing to 256,256
feature_datagen_a = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
feature_datagen_b = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

# pipeline to flow images from a directory in batches
feature_image_generator_a = feature_datagen_a.flow_from_directory('images_a', seed=1, class_mode=None, batch_size = training_batch_size)
feature_image_generator_b = feature_datagen_b.flow_from_directory('images_b', seed=1, class_mode=None, batch_size = training_batch_size)

# core training loop, loads batch of images, generators and discriminators on batch
for i in range(0, number_training_batches):
  print('training batch: ' + str(i))

  # load a batch of images of each class into memory
  images_a_batch = next(feature_image_generator_a)
  images_b_batch = next(feature_image_generator_b)
  target_a_batch = np.ones([len(images_a_batch),1])
  target_b_batch = np.ones([len(images_b_batch),1])

  # fit each generator
  gan_a_to_b.fit(images_a_batch, [target_a_batch, images_a_batch, images_a_batch], batch_size=1)
  gan_b_to_a.fit(images_b_batch, [target_b_batch, images_b_batch, images_b_batch], batch_size=1)

  # create a new set of false images to train discriminator
  images_b_batch_fake = generator_a_to_b.predict(images_a_batch, batch_size=1)
  images_a_batch_fake = generator_b_to_a.predict(images_b_batch, batch_size=1)
  target_a_batch_fake = np.zeros([len(images_a_batch_fake),1])
  target_b_batch_fake = np.zeros([len(images_b_batch_fake),1])

  # combine fake and real images by class
  images_a_batch_discriminator = np.concatenate((images_a_batch, images_a_batch_fake), axis=0)
  images_b_batch_discriminator = np.concatenate((images_b_batch, images_b_batch_fake), axis=0)
  target_a_batch_discriminator = np.concatenate((target_a_batch, target_a_batch_fake), axis=0)
  target_b_batch_discriminator = np.concatenate((target_b_batch, target_b_batch_fake), axis=0)

  # fit discriminator to determine real vs fake images in a class
  discriminator_a.fit(images_a_batch_discriminator, target_a_batch_discriminator, batch_size=1)
  discriminator_b.fit(images_b_batch_discriminator, target_b_batch_discriminator, batch_size=1)

  # create a second training set for the discriminators of all real images mixing the classes
  images_a_batch_discriminator = np.concatenate((images_a_batch, images_b_batch), axis=0)
  images_b_batch_discriminator = np.concatenate((images_b_batch, images_a_batch), axis=0)
  target_a_batch_discriminator = np.concatenate((target_a_batch, target_a_batch_fake), axis=0)
  target_b_batch_discriminator = np.concatenate((target_b_batch, target_b_batch_fake), axis=0)

  # train discriminators to determine real images of class a from real images of class b
  discriminator_a.fit(images_a_batch_discriminator, target_a_batch_discriminator, batch_size=1)
  discriminator_b.fit(images_b_batch_discriminator, target_b_batch_discriminator, batch_size=1)

generator_a_to_b.save('models/generator_a_to_b.h5')
generator_b_to_a.save('models/generator_b_to_a.h5')

discriminator_a.save('models/discriminator_a.h5')
discriminator_b.save('models/discriminator_b.h5')
