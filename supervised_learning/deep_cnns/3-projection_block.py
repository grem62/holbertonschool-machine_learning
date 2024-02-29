#!/usr/bin/env python3
"""
Fonction qui construit un bloc de projection tel que décrit dans Deep Residual Learning
for Image Recognition (2015)
"""
import tensorflow.keras as K


def bloc_projection(A_prev, filtres, s=2):
    """
    Fonction qui construit un bloc de projection tel que décrit dans Deep Residual
    Learning for Image Recognition (2015)
        Arguments :
        - A_prev : la sortie de la couche précédente
        - filtres : un tuple ou une liste contenant F11, F3, F12, respectivement :
            * F11 : le nombre de filtres dans la première convolution 1x1
            * F3 : le nombre de filtres dans la convolution 3x3
            * F12 : le nombre de filtres dans la deuxième convolution 1x1 ainsi que
            la convolution 1x1 dans la connexion shortcut
        - s : la taille du pas de la première convolution à la fois dans le chemin principal
        et la connexion shortcut
        Retourne : la sortie activée du bloc de projection
    """
    F11, F3, F12 = filtres

    # Initialisation He normal est couramment utilisée pour les activations ReLU
    initialiseur = K.initializers.HeNormal(seed=None)

    # Premier composant du chemin principal
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initialiseur
    )(A_prev)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu_1 = K.layers.Activation('relu')(batch_normalization_1)

    # Deuxième composant du chemin principal
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initialiseur
    )(relu_1)
    batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu_2 = K.layers.Activation('relu')(batch_normalization_2)

    # Troisième composant du chemin principal
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initialiseur
    )(relu_2)
    batch_normalization_3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Connexion shortcut
    shortcut_connection = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initialiseur
    )(A_prev)
    batch_normalization_shortcut = K.layers.BatchNormalization(
        axis=3)(shortcut_connection)

    # Ajouter la valeur shortcut à la sortie
    sum_result = K.layers.Add()(
        [batch_normalization_3, batch_normalization_shortcut])

    # Activer la sortie finale
    activated_output = K.layers.Activation(activation='relu')(sum_result)

    return activated_output
