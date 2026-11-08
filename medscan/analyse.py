"""Détection d'anomalies sur des images médicales par analyse des pixels.

Principe : une image est comparée, pixel par pixel, à un modèle « sain » (moyenne et
écart-type de plusieurs images sans anomalie, après lissage). Les pixels qui s'écartent
trop (score z) sont regroupés en régions ; les régions trop petites sont rejetées.
Tout est écrit avec numpy, sans bibliothèque de traitement d'image.
"""
import numpy as np


# --- Outils d'image ---------------------------------------------------------

def otsu(image, bins=256):
    """Seuil d'Otsu : maximise la variance entre les deux classes de l'histogramme."""
    x = np.asarray(image, float).ravel()
    h, bords = np.histogram(x, bins=bins)
    centres = 0.5 * (bords[:-1] + bords[1:])
    p = h / h.sum()
    w0 = np.cumsum(p)
    m = np.cumsum(p * centres)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    var[~np.isfinite(var)] = 0
    k = int(np.argmax(var))
    return bords[k + 1]


def moyenne_locale(image, r):
    """Moyenne sur une fenêtre carrée (2r+1)², par images intégrales (bords répliqués)."""
    x = np.pad(np.asarray(image, float), r, mode="edge")
    s = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


def _decaler(m, dy, dx):
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def eroder(m):
    v = [_decaler(m, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.logical_and.reduce(v)


def dilater(m):
    v = [_decaler(m, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.logical_or.reduce(v)


def ouverture(m):
    """Érosion puis dilatation 3×3 : supprime les pixels isolés, garde les régions."""
    return dilater(eroder(m))


def composantes(m):
    """Étiquetage en 8-connexité par union-find. Renvoie (étiquettes, nombre)."""
    m = np.asarray(m, bool)
    h, w = m.shape
    lab = np.zeros((h, w), int)
    parent = [0]

    def trouver(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for y in range(h):
        for x in range(w):
            if not m[y, x]:
                continue
            voisins = [lab[y + dy, x + dx] for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1))
                       if 0 <= y + dy and 0 <= x + dx < w and lab[y + dy, x + dx]]
            if not voisins:
                parent.append(len(parent))
                lab[y, x] = len(parent) - 1
            else:
                r = min(trouver(v) for v in voisins)
                lab[y, x] = r
                for v in voisins:
                    parent[trouver(v)] = r
    racines = {}
    for y, x in zip(*np.nonzero(lab)):
        r = trouver(lab[y, x])
        lab[y, x] = racines.setdefault(r, len(racines) + 1)
    return lab, len(racines)


# --- Modèle sain et détection ----------------------------------------------

def modele_sain(images_saines, r=2):
    """Moyenne et écart-type pixel par pixel d'images saines lissées."""
    pile = np.stack([moyenne_locale(im, r) for im in images_saines])
    return pile.mean(0), pile.std(0, ddof=1) + 1e-6, r


def detecter(image, modele, seuil_z=5.0, surface_min=12, mm_par_pixel=1.0):
    """Renvoie (masque des anomalies, carte des scores z, liste des régions)."""
    moy, et, r = modele
    z = (moyenne_locale(image, r) - moy) / et
    corps = moy > otsu(moy)
    brut = (np.abs(z) > seuil_z) & corps
    masque = ouverture(brut)
    lab, n = composantes(masque)
    regions = []
    final = np.zeros_like(masque)
    for k in range(1, n + 1):
        zone = lab == k
        aire = int(zone.sum())
        if aire < surface_min:
            continue
        final |= zone
        ys, xs = np.nonzero(zone)
        contraste = float((image[zone] - moy[zone]).mean())
        regions.append({
            "aire_px": aire, "aire_mm2": aire * mm_par_pixel ** 2,
            "centre": (float(ys.mean()), float(xs.mean())),
            "contraste": contraste,
            "type": "hyperintense" if contraste > 0 else "hypointense",
            "z_max": float(np.abs(z[zone]).max()),
        })
    regions.sort(key=lambda d: -d["aire_px"])
    return final, z, regions


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s
