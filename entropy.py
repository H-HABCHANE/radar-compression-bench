"""Limite théorique du taux de compression sans perte, avant tout essai de compresseur.

Pour une source sans mémoire, aucun compresseur sans perte ne peut descendre en moyenne sous
l'entropie H (en bits par échantillon) : le taux maximal est 16 / H pour des échantillons de 16 bits.

  - entropie mesurée (ordre 0) : histogramme des valeurs int16 ;
  - entropie théorique d'un bruit gaussien quantifié d'écart-type s (en unités du bit de poids faible) :
    H = log2(s * sqrt(2 pi e)), valable pour s >= 1 ;
  - entropie après delta d'une impulsion à l'autre : ce qui reste quand on retire ce qui se répète.

Sortie : results/entropy.csv
Exécuter : python entropy.py
"""

import glob
import json
import os

import numpy as np


def entropy_bits(values):
    """Entropie empirique d'ordre 0, en bits par échantillon."""
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    print(f"{'données':12s} {'s par voie':>10s} {'H théorique':>12s} {'H mesurée':>10s} "
          f"{'H après delta':>14s} {'taux max':>9s} {'taux max delta':>15s}")
    for path in sorted(glob.glob("data/*.bin")):
        name = os.path.basename(path)[:-4]
        meta = json.load(open(path[:-4] + ".json"))
        stride = 2 * meta["n_range"]
        x = np.fromfile(path, dtype=np.int16, count=stride * 2048)  # 2048 impulsions suffisent
        s = meta["sigma"] / np.sqrt(2)  # écart-type du bruit sur chaque voie I et Q
        h_theory = np.log2(s * np.sqrt(2 * np.pi * np.e))
        h = entropy_bits(x)
        d = (x[stride:].astype(np.int32) - x[:-stride].astype(np.int32))
        h_delta = entropy_bits(d)
        rows.append((name, s, h_theory, h, h_delta, 16 / h, 16 / h_delta))
        print(f"{name:12s} {s:10.2f} {h_theory:12.3f} {h:10.3f} {h_delta:14.3f} {16 / h:9.2f} {16 / h_delta:15.2f}")
    with open("results/entropy.csv", "w") as f:
        f.write("dataset,sigma_par_voie,h_theorique,h_mesuree,h_apres_delta,taux_max,taux_max_delta\n")
        for r in rows:
            f.write(r[0] + "," + ",".join(f"{v:.5g}" for v in r[1:]) + "\n")


if __name__ == "__main__":
    main()
