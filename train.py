import model_zoo
import cv2
import numpy
from tensorflow import keras

gan_a_to_b, gan_b_to_a, generator_a_to_b, generator_b_to_a, discriminator_a, discriminator_b = model_zoo.create_gans(nodes_per_layer=32)

feature_datagen_a = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

feature_datagen_b = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

feature_image_generator_a = feature_datagen_a.flow_from_directory('images_a', seed=1, class_mode=None, batch_size = 128)
feature_image_generator_b = feature_datagen_b.flow_from_directory('images_b', seed=1, class_mode=None, batch_size = 128)

for i in range(0, 30):
  images_a_batch = next(feature_image_generator_a)
  images_b_batch = next(feature_image_generator_b)
  target_a_batch = np.ones([len(images_a_batch),1])
  target_b_batch = np.ones([len(images_b_batch),1])

  gan_a_to_b.fit(images_a_batch, [target_a_batch, images_a_batch, images_a_batch], batch_size=16)
  gan_b_to_a.fit(images_b_batch, [target_b_batch, images_b_batch, images_b_batch], batch_size=16)

  images_b_batch_fake = generator_a_to_b.predict(images_a_batch, batch_size=16)
  images_a_batch_fake = generator_b_to_a.predict(images_b_batch, batch_size=16)
  target_a_batch_fake = np.zeros([len(images_a_batch_fake),1])
  target_b_batch_fake = np.zeros([len(images_b_batch_fake),1])

  images_a_batch_discriminator = np.concatenate((images_a_batch, images_a_batch_fake), axis=0)
  images_b_batch_discriminator = np.concatenate((images_b_batch, images_b_batch_fake), axis=0)
  target_a_batch_discriminator = np.concatenate((target_a_batch, target_a_batch_fake), axis=0)
  target_b_batch_discriminator = np.concatenate((target_b_batch, target_b_batch_fake), axis=0)

  discriminator_a.fit(images_a_batch_discriminator, target_a_batch_discriminator)
  discriminator_b.fit(images_b_batch_discriminator, target_b_batch_discriminator)
  
i=0
for batch in feature_image_generator_b:
  i += 1
  print(i*128)
  if i * 128 > 2000:
    break
  fake_batch = generator_b_to_a.predict(batch)
  for image in fake_batch[np.where(discriminator_a.predict(fake_batch)[:,0] > 0.8)]:
    cv2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

i=0
for batch in feature_image_generator_a:
  i +=1
  print(i*128)
  if i * 128 > 2000:
    break
  fake_batch = generator_a_to_b.predict(batch)
  for image in fake_batch[np.where(discriminator_b.predict(fake_batch)[:,0] > 0.8)]:
    cv2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
