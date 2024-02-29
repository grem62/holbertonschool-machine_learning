#!/usr/bin/env python3
"""sumary"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    input = K.Input(shape=(224, 224, 3))
    
    # Stage 1
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            strides=(2, 2), padding='same',
                            kernel_initializer='he_normal')(input)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2), padding='same')(relu1)
    
    # Stage 2
    proj2 = projection_block(pool1, [64, 64, 256], 1)
    iden2_1 = identity_block(proj2, [64, 64, 256])
    iden2_2 = identity_block(iden2_1, [64, 64, 256])
    
    # Stage 3
    proj3 = projection_block(iden2_2, [128, 128, 512])
    iden3_1 = identity_block(proj3, [128, 128, 512])
    iden3_2 = identity_block(iden3_1, [128, 128, 512])
    iden3_3 = identity_block(iden3_2, [128, 128, 512])
    
    # Stage 4
    proj4 = projection_block(iden3_3, [256, 256, 1024])
    iden4_1 = identity_block(proj4, [256, 256, 1024])
    iden4_2 = identity_block(iden4_1, [256, 256, 1024])
    iden4_3 = identity_block(iden4_2, [256, 256, 1024])
    iden4_4 = identity_block(iden4_3, [256, 256, 1024])
    iden4_5 = identity_block(iden4_4, [256, 256, 1024])
    
    # Stage 5
    proj5 = projection_block(iden4_5, [512, 512, 2048])
    iden5_1 = identity_block(proj5, [512, 512, 2048])
    iden5_2 = identity_block(iden5_1, [512, 512, 2048])
    
    # Average pooling and output layer
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1))(iden5_2)
    output = K.layers.Dense(units=1000, activation='softmax')(avg_pool)
    
    model = K.Model(inputs=input, outputs=output)
    
    return model
