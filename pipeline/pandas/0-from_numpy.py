#!/usr/bin/env python3

"import panda"

import pandas as pd
import string


def from_numpy(array):
    # Générer les noms de colonnes de 'A' à 'Z'
    num_cols = array.shape[1]
    column_labels = list(string.ascii_uppercase[:num_cols])

    # Créer le DataFrame
    df = pd.DataFrame(array, columns=column_labels)

    return df
