#!/usr/bin/env python3
""" Optimiser k """

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Teste le nombre optimal de clusters en utilisant la variance
    Arguments :
        - X est un tableau numpy de forme (n, d)
        contenant l'ensemble de données
        - kmin est un entier positif contenant
        le nombre minimum de clusters
            à vérifier (inclusif)
        - kmax est un entier positif contenant
        le nombre maximum de clusters
            à vérifier (inclusif)
        - iterations est un entier positif
        contenant le nombre maximum d'itérations
            pour K-means
    Retourne : results, d_vars, ou None, None en cas d'échec
        - results est une liste contenant les sorties
        de K-means pour chaque taille de cluster
        - d_vars est une liste contenant la différence
        de variance par rapport à la
            plus petite taille de cluster pour
            chaque taille de cluster
    """
    # Vérification si X est un tableau numpy avec deux dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    # Vérification si kmin est un entier positif
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    # Vérification si kmax est un entier positif
    if kmax is not None and (not isinstance(kmax, int) or kmax < 1):
        return None, None
    # Vérification si kmax est supérieur à kmin
    if kmax is not None and kmax <= kmin:
        return None, None
    # Vérification si iterations est un entier positif
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    # Extraction du nombre de points de données
    # (n) et de dimensions (d) à partir de X
    n, d = X.shape
    # Définition de kmax à n s'il est None
    if kmax is None:
        kmax = n
    # Initialisation des listes results et d_vars
    results = []
    d_vars = []
    # Boucle pour itérer à travers la plage de clusters
    for k in range(kmin, (kmax or kmin) + 1):
        # Exécution de K-means sur l'ensemble de données
        C, clss = kmeans(X, k, iterations)
        # Vérification si K-means a échoué
        results.append((C, clss))
        # Calcul de la variance pour la plus petite taille de cluster
        if k == kmin:
            var_min = variance(X, C)
        # Calcul de la différence de variance par rapport
        # à la plus petite taille de cluster
        d_vars.append(var_min - variance(X, C))
    # Retourne les résultats et la différence de variance
    return results, d_vars
