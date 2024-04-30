#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os  # Ajout de la bibliothèque os pour manipuler les chemins de fichiers

# Obtenez le chemin absolu du répertoire courant du script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Mettez à jour le chemin du fichier en utilisant os.path.join pour garantir la portabilité entre les systèmes d'exploitation
chemin_train = os.path.join(script_directory, 'data', 'Binary_Train.npz')

# Chargez les données depuis le fichier
lib_train = np.load(chemin_train)
X_3D, Y = lib_train['X'], lib_train['Y']

# Reste du code inchangé
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
