#!/usr/bin/env python3
"autoencoder variational"

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from keras import backend as K

def autoencoder(input_dims, hidden_layers, latent_dims):
    inputs = K.Input(shape=(input_dims,))
    h = layers.Dense(hidden_layers, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dims)(h)
    z_log_sigma = layers.Dense(latent_dims)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims),
                              mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = K.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = K.Input(shape=(latent_dims,), name='z_sampling')
    x = layers.Dense(hidden_layers, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = K.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = K.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return encoder, decoder, vae
