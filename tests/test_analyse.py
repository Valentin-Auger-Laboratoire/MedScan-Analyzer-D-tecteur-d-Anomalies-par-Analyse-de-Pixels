"""Vérifications : python tests/test_analyse.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from medscan import (otsu, moyenne_locale, ouverture, composantes, modele_sain, detecter,
                     dice, coupe, sujets_sains)

n_ok = 0


def ok(cond, msg):
    global n_ok
    if not cond:
        raise AssertionError(msg)
    n_ok += 1


rng = np.random.default_rng(0)

# Otsu : sépare deux populations, et coïncide avec la recherche exhaustive
x = np.r_[rng.normal(50, 8, 5000), rng.normal(150, 10, 3000)]
s = otsu(x)
ok(70 < s < 130, f"Otsu entre les deux modes ({s:.1f})")
def var_inter(t):
    a, b = x[x < t], x[x >= t]
    return len(a) * len(b) * (a.mean() - b.mean()) ** 2
brute = max(np.linspace(60, 140, 801), key=var_inter)
ok(abs(s - brute) < 1.0, "Otsu = maximum de la variance inter-classes")

# Moyenne locale : identique au calcul direct
im = rng.normal(0, 1, (20, 23))
for r in (1, 2):
    p = np.pad(im, r, mode="edge")
    direct = np.array([[p[y:y + 2 * r + 1, x:x + 2 * r + 1].mean() for x in range(23)] for y in range(20)])
    ok(np.allclose(moyenne_locale(im, r), direct), f"moyenne locale r={r}")

# Composantes connexes et ouverture
m = np.zeros((10, 12), bool)
m[1:3, 1:3] = True; m[5:8, 6:10] = True; m[3, 3] = True  # le pixel (3,3) touche le premier bloc en diagonale
lab, n = composantes(m)
ok(n == 2 and lab[3, 3] == lab[1, 1], "8-connexité : 2 régions")
m2 = np.zeros((10, 10), bool); m2[2:7, 2:7] = True; m2[0, 9] = True
o = ouverture(m2)
ok(not o[0, 9] and o[2:7, 2:7].all(), "l'ouverture retire le pixel isolé et garde le carré")
u = np.zeros((6, 9), bool); u[1:5, 1] = u[1:5, 7] = u[4, 1:8] = True  # forme en U : une seule région
ok(composantes(u)[1] == 1, "forme en U = une seule région (fusion des étiquettes)")

# Détection sur des coupes fictives
modele = modele_sain(sujets_sains(20))
for g in range(8):  # aucune fausse alerte sur des sujets sains jamais vus
    im, _ = coupe(graine=500 + g, decalage=rng.uniform(-0.5, 0.5))
    ok(len(detecter(im, modele)[2]) == 0, f"pas de fausse alerte (sujet sain {g})")
for g, (cy, cx, r, c) in enumerate([(50, 45, 6, 35), (80, 82, 4, -30), (64, 40, 5, 25), (45, 80, 3, 40)]):
    im, vrai = coupe(lesions=[(cy, cx, r, c)], graine=700 + g)
    masque, z, reg = detecter(im, modele)
    ok(len(reg) == 1, f"lésion {g} détectée une seule fois")
    ok(abs(reg[0]["centre"][0] - cy) < 1.5 and abs(reg[0]["centre"][1] - cx) < 1.5, f"lésion {g} bien localisée")
    ok(reg[0]["type"] == ("hyperintense" if c > 0 else "hypointense"), f"lésion {g} bon type")
    ok(dice(masque, vrai) > 0.6, f"lésion {g} : Dice {dice(masque, vrai):.2f}")
im, vrai = coupe(lesions=[(50, 45, 6, 35), (80, 82, 4, -30)], graine=7)
ok(len(detecter(im, modele)[2]) == 2, "deux lésions dans la même coupe")

# Limite de détection : une lésion minuscule et peu contrastée passe inaperçue (et c'est dit)
im, _ = coupe(lesions=[(60, 60, 1.5, 8)], graine=9)
ok(len(detecter(im, modele)[2]) == 0, "lésion de 1,5 px à +8 niveaux : sous la limite de détection")

ok(dice(np.ones((3, 3)), np.ones((3, 3))) == 1 and dice(np.zeros(4), np.zeros(4)) == 1, "Dice")

print(f"{n_ok} vérifications réussies")
