import tensorflow.keras as K
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Input, Lambda, Dense, Flatten, MaxPooling2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

def preprocess_data(X, Y):
    """Preprocesses the CIFAR-10 data."""
    X_p = K.applications.resnet_v2.preprocess_input(X)
    Y_p = to_categorical(Y, num_classes=10)
    return X_p, Y_p

def main():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess the data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Define ResNet50V2 model
    base_model = K.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),  # Updated input shape to match resized image
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    input_tensor = Input(shape=(32, 32, 3))
    x = Lambda(lambda image: K.backend.resize_images(image, 224, 224, "channels_last"))(input_tensor)
    x = base_model(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Add MaxPooling2D layer
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print("Validation accuracy: {:.2f}%".format(accuracy * 100))

    # Save the model
    model.save('cifar10.h5'.K)

if __name__ == "__main__":
    main()
