"""Images de synthèse : une coupe de « crâne » simplifiée, avec ou sans lésions.

Toutes les images sont fictives (ellipses, bruit gaussien) : aucune donnée de patient.
"""
import numpy as np

TAILLE = 128


def _ellipse(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def coupe(lesions=(), bruit=6.0, graine=0, taille=TAILLE, decalage=0.0):
    """Renvoie (image en niveaux 0–255, masque vrai des lésions).

    lesions : liste de (cy, cx, rayon, contraste) ; contraste en niveaux de gris,
    positif (hyperintense) ou négatif (hypointense). `decalage` : petite variation
    d'anatomie (en pixels) d'un sujet sain à l'autre.
    """
    rng = np.random.default_rng(graine)
    yy, xx = np.mgrid[0:taille, 0:taille].astype(float)
    c = taille / 2
    im = np.full((taille, taille), 10.0)
    im[_ellipse(yy, xx, c, c, 0.42 * taille, 0.36 * taille)] = 200  # os du crâne
    im[_ellipse(yy, xx, c, c, 0.39 * taille, 0.33 * taille)] = 110  # cerveau
    for sx in (-1, 1):  # ventricules
        im[_ellipse(yy, xx, c - 4, c + sx * (6 + decalage), 9, 4)] = 40
    vrai = np.zeros((taille, taille), bool)
    for cy, cx, r, contraste in lesions:
        disque = _ellipse(yy, xx, cy, cx, r, r)
        im[disque] += contraste
        vrai |= disque
    im += bruit * rng.standard_normal(im.shape)
    return np.clip(im, 0, 255), vrai


def sujets_sains(n=20, graine=100, **kw):
    """Plusieurs coupes saines, avec une petite variabilité d'anatomie."""
    rng = np.random.default_rng(graine)
    return [coupe(graine=graine + i, decalage=rng.uniform(-0.5, 0.5), **kw)[0] for i in range(n)]
