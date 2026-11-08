"""Démonstration : crée une coupe fictive avec deux lésions, les détecte et écrit l'image annotée.

Usage : python analyse.py [--seuil 5] [--sortie images/resultat.png]
"""
import argparse

import numpy as np
from PIL import Image

from medscan import coupe, sujets_sains, modele_sain, detecter, dice, dilater


def annoter(image, masque):
    rgb = np.repeat(np.asarray(image, np.uint8)[..., None], 3, axis=2)
    contour = dilater(masque) & ~masque
    rgb[contour] = (200, 16, 46)
    return Image.fromarray(rgb).resize((image.shape[1] * 3, image.shape[0] * 3), Image.NEAREST)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seuil", type=float, default=5.0, help="score z au-delà duquel un pixel est anormal")
    p.add_argument("--sortie", default="images/resultat.png")
    a = p.parse_args()

    lesions = [(50, 45, 6, 35), (80, 82, 4, -30)]
    image, vrai = coupe(lesions=lesions, graine=7)
    modele = modele_sain(sujets_sains(20))
    masque, z, regions = detecter(image, modele, seuil_z=a.seuil, mm_par_pixel=1.5)
    print(f"{len(regions)} anomalie(s) détectée(s) (seuil z = {a.seuil:g}) :")
    for i, r in enumerate(regions, 1):
        cy, cx = r["centre"]
        print(f"  {i}. {r['type']:12s} centre ({cy:.0f}, {cx:.0f}) · {r['aire_px']} px = {r['aire_mm2']:.0f} mm² · "
              f"contraste {r['contraste']:+.0f} · z max {r['z_max']:.1f}")
    print(f"Recouvrement avec la vérité (Dice) : {dice(masque, vrai):.2f}")
    annoter(image, masque).save(a.sortie)
    print(f"Image annotée : {a.sortie}")


if __name__ == "__main__":
    main()
