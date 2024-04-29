#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

# Obtenez le chemin absolu du répertoire courant du script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Mettez à jour le chemin du fichier en utilisant os.path.join
chemin_mnist = os.path.join(script_directory, 'data', 'MNIST.npz')

# Chargez les données depuis le fichier
lib = np.load(chemin_mnist)
print(lib.files)
X_train_3D = lib['X_train']
Y_train = lib['Y_train']

# Reste du code inchangé
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_train_3D[i])
    plt.title(str(Y_train[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

