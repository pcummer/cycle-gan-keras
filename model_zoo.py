from tensorflow import keras

def generator(nodes_per_layer=32):
    '''Build a deep generative model that takes an image and returns an image
    :param nodes_per_layer:
    :return: model
    '''
    input_x = keras.layers.Input(shape=(256, 256, 3))

    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(input_x)
    x = keras.layers.Activation('relu')(x)

    # three residual blocks working at the full dimension
    res_x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(x)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x = keras.layers.Add()([x, res_x])

    # six residual blocks working at half the full dimension
    x_half = keras.layers.Conv2D(nodes_per_layer * 2, (2, 2), (2, 2))(input_x)
    x_half = keras.layers.Activation('relu')(x_half)

    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_half = keras.layers.Add()([x_half, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_half = keras.layers.Add()([x_half, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_half = keras.layers.Add()([x_half, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_half = keras.layers.Add()([x_half, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_half = keras.layers.Add()([x_half, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_half = keras.layers.Add()([x_half, res_x])

    x_quarter = keras.layers.Conv2D(nodes_per_layer * 4, (2, 2), (2, 2))(x_half)
    x_quarter = keras.layers.Activation('relu')(x_quarter)

    # six residual blocks working at one quarter of the full dimension
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_quarter = keras.layers.Add()([x_quarter, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_quarter = keras.layers.Add()([x_quarter, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_quarter = keras.layers.Add()([x_quarter, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_quarter = keras.layers.Add()([x_quarter, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_quarter = keras.layers.Add()([x_quarter, res_x])

    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
    res_x = keras.layers.Activation('relu')(res_x)
    res_x = keras.layers.Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
    res_x = keras.layers.Activation('relu')(res_x)
    x_quarter = keras.layers.Add()([x_quarter, res_x])

    x_quarter = keras.layers.Conv2DTranspose(nodes_per_layer * 2, (2, 2), (2, 2))(x_quarter)
    x_quarter = keras.layers.Activation('relu')(x_quarter)

    x_half = keras.layers.Add()([x_quarter, x_half])

    x_half = keras.layers.Conv2DTranspose(nodes_per_layer, (2, 2), (2, 2))(x_half)
    x_half = keras.layers.Activation('relu')(x_half)

    x = keras.layers.Add()([x, x_half])
    x = keras.layers.Conv2D(3, (3, 3), padding='same')(x)
    output_x = keras.layers.Activation('relu')(x)
    return keras.Model(input_x, output_x)

def discriminator(nodes_per_layer=32):
    '''Discriminator to determine if an image is a real example of a class
    :param nodes_per_layer:
    :return: model
    '''
    input_x = keras.layers.Input(shape=(256, 256, 3))

    # downsampling is acomplished with strided convolution rather than pooling to better preserve gradients
    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(input_x)
    x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.Conv2D(nodes_per_layer, (3, 3), (2, 2))(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    # aggressive dropout to force the model to evaluate different spatial patches
    x = keras.layers.Dropout(0.5)(x)

    # essential average with learned weights since a true average would not train
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)
    output_x = keras.layers.Activation('sigmoid')(x)
    return keras.Model(input_x, output_x)

def create_gans(nodes_per_layer=32):
    '''Shares generators and discriminators to create two paired GANs with appropriate weight sharing
    :param nodes_per_layer:
    :return: model, model, model, model, model, model
    '''
    # create building blocks
    generator_a_to_b = generator(nodes_per_layer=nodes_per_layer)
    generator_b_to_a = generator(nodes_per_layer=nodes_per_layer)
    discriminator_a = discriminator(nodes_per_layer=nodes_per_layer)
    discriminator_b = discriminator(nodes_per_layer=nodes_per_layer)

    # compile discriminators while they're set to trainable
    slow_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_a.compile(optimizer=slow_optimizer, loss='binary_crossentropy', metrics=['acc'])
    slow_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_b.compile(optimizer=slow_optimizer, loss='binary_crossentropy', metrics=['acc'])

    # freeze discriminators so they will not train within the full GAN
    discriminator_a.trainable = False
    discriminator_b.trainable = False

    # create the first GAN architecture
    input_a = keras.layers.Input(shape=(256, 256, 3))
    output_image_b = generator_a_to_b(input_a)
    output_discriminator_b = discriminator_b(output_image_b)
    output_reconstruction_a = generator_b_to_a(output_image_b)

    # create the second GAN architecture
    input_b = keras.layers.Input(shape=(256, 256, 3))
    output_image_a = generator_b_to_a(input_b)
    output_discriminator_a = discriminator_a(output_image_a)
    output_reconstruction_b = generator_a_to_b(output_image_a)

    # assemble and compile models
    gan_a_to_b = keras.Model(inputs=input_a, outputs=[output_discriminator_b, output_reconstruction_a, output_image_b])
    gan_b_to_a = keras.Model(inputs=input_b, outputs=[output_discriminator_a, output_reconstruction_b, output_image_a])

    slow_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    gan_a_to_b.compile(optimizer=slow_optimizer, loss=['binary_crossentropy', 'mae', 'mae'],
                       loss_weights=[0.1, 1, 0.01])
    slow_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    gan_b_to_a.compile(optimizer=slow_optimizer, loss=['binary_crossentropy', 'mae', 'mae'],
                       loss_weights=[0.1, 1, 0.01])
    return gan_a_to_b, gan_b_to_a, generator_a_to_b, generator_b_to_a, discriminator_a, discriminator_b
