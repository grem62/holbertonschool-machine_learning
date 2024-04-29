#!/usr/bin/env python3
"""
Ce script est destiné à être exécuté avec Python 3.
"""

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """
    Cette classe définit un seul neurone effectuant une classification binaire.
    """

    def __init__(self, nx):
        """
        Initialisation d'un neurone.

        Args:
            nx (int): nombre de caractéristiques d'entrée pour le neurone.
        """
        # Vérification des paramètres d'entrée
        if type(nx) is not int:
            raise TypeError('nx doit être un entier')
        if nx < 1:
            raise ValueError('nx doit être un entier positif')

        # Initialisation des poids (W), du biais (b) et de l'activation (A)
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Fonction getter pour l'instance privée W."""
        return self.__W

    @property
    def b(self):
        """Fonction getter pour l'instance privée b."""
        return self.__b

    @property
    def A(self):
        """Fonction getter pour l'instance privée A."""
        return self.__A

    def forward_prop(self, X):
        """
        Propagation avant du neurone.

        Args:
            X (numpy array): données d'entrée.

        Returns:
            numpy array: sortie activée du neurone (prédiction).
        """
        # Calcul de l'activation (A) en utilisant la fonction sigmoïde
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A
        """Fonction pour calculer la propagation avant du neurone"""

    def cost(self, Y, A):
        """
        Fonction de coût pour évaluer les performances du modèle.

        Args:
            Y (numpy array): étiquettes réelles.
            A (numpy array): prédictions du modèle.

        Returns:
            float: coût.
        """
        # Calcul du coût en utilisant la fonction de coût logistique
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        """Fonction mathématique pour calculer le coût du modèle."""
        return cost

    def evaluate(self, X, Y):
        """
        Évalue les performances du neurone.

        Args:
            X (numpy array): données d'entrée.
            Y (numpy array): étiquettes réelles.

        Returns:
            Tuple: (prédictions du modèle arrondies, coût).
        """
        # Obtenir les prédictions et le coût en utilisant les données d'entrée
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.round(A).astype(int), cost
        """Fonction pour évaluer le neurone"""

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Effectue la descente de gradient pour mettre à jour les poids du neurone.

        Args:
            X (numpy array): données d'entrée.
            Y (numpy array): étiquettes réelles.
            A (numpy array): prédictions du modèle.
            alpha (float, facultatif): taux d'apprentissage. Par défaut 0.05.
        """
        # Calcul des gradients par rapport aux poids (dW) et au biais (db)
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        # Mise à jour des poids et du biais en utilisant la descente de gradient
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
        """Fonction pour calculer la descente de gradient pour le neurone"""

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Entraîne le neurone en utilisant la descente de gradient.

        Args:
            X (numpy array): données d'entrée.
            Y (numpy array): étiquettes réelles.
            iterations (int, facultatif): nombre d'itérations d'entraînement. Par défaut 5000.
            alpha (float, facultatif): taux d'apprentissage. Par défaut 0.05.

        Returns:
            Tuple: (prédictions du modèle arrondies, coût).
        """
        # Vérification des paramètres d'entrée
        if type(iterations) is not int:
            raise TypeError('iterations doit être un entier')
        if iterations < 0:
            raise ValueError('iterations doit être un entier positif')
        if type(alpha) is not float:
            raise TypeError('alpha doit être un flottant')
        if alpha < 0:
            raise ValueError('alpha doit être positif')
        if type(step) is not int:
            raise TypeError('step doit être un entier')
        if step <= 0 or step > iterations:
            raise ValueError('step doit être positif et <= iterations')

        # Initialisation des listes pour stocker les coûts et les itérations
        costs = []
        list_iterations = []

        # Boucle d'entraînement sur le nombre d'itérations spécifié
        for i in range(iterations + 1):
            # Effectuer la descente de gradient pour mettre à jour les poids
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
            # Calculer le coût actuel
            cost = self.cost(Y, self.__A)
            costs.append(cost)
            list_iterations.append(i)

            # Affichage du coût à chaque itération spécifiée
            if verbose and i % step == 0:
                print('cost {} iterations : {}'.format(i, costs[i]))

        # Affichage du graphique du coût si spécifié
        if graph:
            plt.plot(costs, list_iterations)
            plt.xlabel('itération')
            plt.ylabel('coût')
            plt.title('Coût d\'entraînement')
            plt.show()

        # Retourner les prédictions et le coût après l'entraînement
        return self.evaluate(X, Y)

