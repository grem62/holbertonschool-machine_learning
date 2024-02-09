import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape  # Note: c_prev est redondant ici car déjà défini
    sh, sw = stride

    # Calcul du padding si nécessaire
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Application du padding sur A_prev si nécessaire
    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Pour chaque élément dans le volume de sortie
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Sélection de la tranche pour dA_prev
                slice_A = A_prev_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                # Mise à jour de dA_prev, dW, et db
                dA_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] += W[:, :, :, k] * dZ[:, i, j, k][:, None, None, None]
                dW[:, :, :, k] += np.sum(slice_A * dZ[:, i, j, k][:, None, None, None], axis=0)
                db[:, :, :, k] += np.sum(dZ[:, i, j, k], axis=0)

    # Ajustement si padding == 'same' pour dA_prev
    if padding == 'same':
        if ph != 0:
            dA_prev = dA_prev[:, ph:-ph, :, :]
        if pw != 0:
            dA_prev = dA_prev[:, :, pw:-pw, :]

    return dA_prev, dW, db
