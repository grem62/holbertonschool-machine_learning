#!/usr/bin/env python3
"autoencoder variational"

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from keras import backend as K

def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    # Encoder
    input_encoder = K.Input(shape=(input_dims,))
    output_encoder = input_encoder
    for units in hidden_layers:
        output_encoder = K.layers.Dense(units, activation='relu')(output_encoder)
    z_mean = K.layers.Dense(latent_dims)(output_encoder)
    z_log_sigma = K.layers.Dense(latent_dims)(output_encoder)

    def sampling(args):
        z_mean, z_log_sigma = args
        batch = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch, latent_dims), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon

    z = K.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = K.Model(input_encoder, [z, z_mean, z_log_sigma])

    # Decoder
    input_decoder = K.Input(shape=(latent_dims,))
    output_decoder = input_decoder
    for units in reversed(hidden_layers):
        output_decoder = K.layers.Dense(units, activation='relu')(output_decoder)
    output_decoder = K.layers.Dense(input_dims, activation='sigmoid')(output_decoder)
    decoder = K.Model(input_decoder, output_decoder)

    # Autoencoder
    output_autoencoder = decoder(encoder(input_encoder)[0])
    autoencoder = K.Model(input_encoder, output_autoencoder)

    def vae_loss(x, x_decoded_mean):
        xent_loss = K.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    autoencoder.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, autoencoder
