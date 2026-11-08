"""Génère images/medscan.png (nécessite matplotlib) : python figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medscan import coupe, sujets_sains, modele_sain, detecter, dilater

image, vrai = coupe(lesions=[(50, 45, 6, 35), (80, 82, 4, -30)], graine=7)
saines = sujets_sains(20)
modele = modele_sain(saines)
masque, z, regions = detecter(image, modele)
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(saines[0], cmap="gray", vmin=0, vmax=255); ax[0].set_title("Un sujet sain (modèle : 20 coupes)")
ax[1].imshow(image, cmap="gray", vmin=0, vmax=255); ax[1].set_title("Coupe à analyser")
im = ax[2].imshow(np.clip(z, -15, 15), cmap="RdBu_r", vmin=-15, vmax=15); ax[2].set_title("Score z par pixel")
fig.colorbar(im, ax=ax[2], fraction=.046)
rgb = np.repeat((image / 255)[..., None], 3, 2)
rgb[dilater(masque) & ~masque] = (0.78, 0.06, 0.18)
ax[3].imshow(rgb); ax[3].set_title(f"{len(regions)} anomalies détectées")
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig("images/medscan.png", dpi=110)
