#!/usr/bin/env python3
""" Convolutional Autoencoder """

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    # Encoder
    input_img = K.Input(shape=input_dims)
    encoded = input_img
    for num_filters in filters:
        encoded = K.layers.Conv2D(num_filters,
                                  (3, 3), activation='relu',
                                  padding='same')(encoded)
        encoded = K.layers.MaxPooling2D((2, 2),
                                        padding='same')(encoded)

    # Latent space
    latent_dims = encoded

    # Decoder
    for num_filters in reversed(filters[:-1]):
        decoded = K.layers.Conv2D(num_filters, (3, 3),
                                  activation='relu',
                                  padding='same')(latent_dims)
        decoded = K.layers.UpSampling2D((2, 2))(decoded)

    decoded = K.layers.Conv2D(filters[-1], (3, 3),
                              activation='relu',
                              padding='valid')(decoded)
    decoded = K.layers.UpSampling2D((2, 2))(decoded)
    decoded = K.layers.Conv2D(input_dims[-1], (3, 3),
                              activation='sigmoid',
                              padding='same')(decoded)

    # Models
    encoder = K.Model(input_img, latent_dims)
    decoder = K.Model(latent_dims, decoded)
    auto = K.Model(input_img, decoder(encoder(input_img)))

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
