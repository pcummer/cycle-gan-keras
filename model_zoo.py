from tensorflow import keras

def generator(nodes_per_layer=32):
    input_x = keras.layers.Input(shape=(256, 256, 3))
    x = keras.layers.Conv2D(nodes_per_layer*2, (2, 2), (2, 2))(input_x)
    x_skip = keras.layers.Activation('relu')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(nodes_per_layer*4, (2, 2), (2, 2))(x_skip)
    x = keras.layers.Activation('relu')(x)
    x_skip = keras.layers.Conv2D(nodes_per_layer*2, (3, 3), padding='same')(x_skip)
    x_skip = keras.layers.Activation('relu')(x_skip)

    x_skip_2 = keras.layers.Conv2D(nodes_per_layer*4, (3, 3), padding='same')(x)
    x_skip_2 = keras.layers.Activation('relu')(x_skip_2)
    

    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])
    
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer*4, (3,3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    x = keras.layers.Add()([x, x_skip_2])
    
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2DTranspose(nodes_per_layer*2, (2, 2), (2, 2))(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Add()([x, x_skip])
    x = keras.layers.Conv2DTranspose(nodes_per_layer, (2, 2), (2, 2))(x)
    x = keras.layers.Activation('relu')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2DTranspose(3, (2, 2), padding='same')(x)
    output_x = keras.layers.Activation('relu')(x)
    return keras.Model(input_x, output_x)

def discriminator(nodes_per_layer=32):
    input_x = keras.layers.Input(shape=(256, 256, 3))
    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(input_x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(1, (10,10))(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)
    output_x = keras.layers.Activation('sigmoid')(x)
    # output_x = keras.layers.Lambda(lambda x: keras.backend.mean(x, keepdims=True))(x)
    return keras.Model(input_x, output_x)

def create_gans(nodes_per_layer=32):
    generator_a_to_b = generator(nodes_per_layer=nodes_per_layer)
    generator_b_to_a = generator(nodes_per_layer=nodes_per_layer)
    discriminator_a = discriminator(nodes_per_layer=nodes_per_layer)
    discriminator_b = discriminator(nodes_per_layer=nodes_per_layer)

    slow_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_a.compile(optimizer=slow_optimizer, loss='binary_crossentropy', metrics=['acc'])
    slow_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_b.compile(optimizer=slow_optimizer, loss='binary_crossentropy', metrics=['acc'])

    discriminator_a.trainable = False
    discriminator_b.trainable = False

    input_a = keras.layers.Input(shape=(256, 256, 3))
    output_image_b = generator_a_to_b(input_a)
    output_discriminator_b = discriminator_b(output_image_b)
    output_reconstruction_a = generator_b_to_a(output_image_b)

    input_b = keras.layers.Input(shape=(256, 256, 3))
    output_image_a = generator_b_to_a(input_b)
    output_discriminator_a = discriminator_a(output_image_a)
    output_reconstruction_b = generator_a_to_b(output_image_a)

    gan_a_to_b = keras.Model(inputs=input_a, outputs=[output_discriminator_b, output_reconstruction_a, output_image_b])
    gan_b_to_a = keras.Model(inputs=input_b, outputs=[output_discriminator_a, output_reconstruction_b, output_image_a])

    gan_a_to_b.compile(optimizer='adam', loss=['binary_crossentropy', 'mae', 'mae'], loss_weights = [0.3, 1, 0.1])
    gan_b_to_a.compile(optimizer='adam', loss=['binary_crossentropy', 'mae', 'mae'], loss_weights = [0.3, 1, 0.1])
    return gan_a_to_b, gan_b_to_a, generator_a_to_b, generator_b_to_a, discriminator_a, discriminator_b
