#!/usr/bin/env python3
""" Convolutional Autoencoder """

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    Arguments:
        - input_dims is a tuple of integers containing the dimensions of the
          model input
        - filters is a list containing the number of filters for each
          convolutional layer in the encoder, respectively
            - the filters should be reversed for the decoder
        - latent_dims is a tuple of integers containing the dimensions of the
          latent space representation
    Returns: encoder, decoder, auto
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the full autoencoder model
    """
    # Encoder
    encoder_input = K.Input(shape=input_dims)
    enc = encoder_input
    for f in filters:
        enc = K.layers.Conv2D(f, (3, 3),
                              activation='relu',
                              padding='same')(enc)
        enc = K.layers.MaxPooling2D((2, 2),
                                    padding='same')(enc)
    encoder = K.models.Model(encoder_input, enc)

    # Decoder
    decoder_input = K.Input(shape=latent_dims)
    dec = decoder_input
    for f in reversed(filters[:-1]):
        dec = K.layers.Conv2D(f, (3, 3),
                              activation='relu',
                              padding='same')(dec)
        dec = K.layers.UpSampling2D((2, 2))(dec)
    dec = K.layers.Conv2D(-1, (3, 3),
                          activation='relu', padding='valid')(dec)
    dec = K.layers.UpSampling2D((2, 2))(dec)
    dec = K.layers.Conv2D(1, (3, 3),
                          activation='sigmoid', padding='same')(dec)
    decoder = K.models.Model(decoder_input, dec)

    # Autoencoder
    auto = K.models.Model(encoder_input, decoder(encoder(encoder_input)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
