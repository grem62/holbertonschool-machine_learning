

import tensorflow.keras as K


def densenet121(growth_rate=32, compression=1.0):
    def dense_block(X, growth_rate, num_layers):
        for _ in range(num_layers):
            # Batch Normalization
            X = K.layers.BatchNormalization()(X)
            # ReLU activation
            X = K.layers.Activation('relu')(X)
            # Convolution
            X = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=1,
                                padding='same',
                                kernel_initializer='he_normal')(X)
            # Batch Normalization
            X = K.layers.BatchNormalization()(X)
            # ReLU activation
            X = K.layers.Activation('relu')(X)
            # Convolution
            X = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3, padding='same',
                                kernel_initializer='he_normal')(X)
            # Concatenate with input
            X = K.layers.concatenate([X, X_input])
            # Update input
            X_input = X
        return X

    def transition_layer(X, compression):
        # Batch Normalization
        X = K.layers.BatchNormalization()(X)
        # ReLU activation
        X = K.layers.Activation('relu')(X)
        # Convolution
        num_filters = int(compression * X.shape[-1])
        X = K.layers.Conv2D(filters=num_filters, kernel_size=1,
                            padding='same', kernel_initializer='he_normal')(X)
        # Average Pooling
        X = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                      padding='valid')(X)
        return X

    # Input layer
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Convolution
    X = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Dense Block 1
    X = K.layers.dense_block(X, growth_rate, num_layers=6)
    X = K.layers.transition_layer(X, compression)

    # Dense Block 2
    X = K.layers.dense_block(X, growth_rate, num_layers=12)
    X = K.layers.transition_layer(X, compression)

    # Dense Block 3
    X = K.layers.dense_block(X, growth_rate, num_layers=24)
    X = K.layers.transition_layer(X, compression)

    # Dense Block 4
    X = K.layers.dense_block(X, growth_rate, num_layers=16)

    # Global Average Pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Output layer
    X = K.layers.Dense(units=1000, activation='softmax')(X)

    # Create the model
    model = K.Model(inputs=X_input, outputs=X)

    return model
