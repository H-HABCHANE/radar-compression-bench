"""Génère des données radar brutes synthétiques : échantillons I/Q d'un convertisseur 14 bits,
stockés en int16 entrelacés (I0, Q0, I1, Q1, ...), rangés par impulsion puis par case distance.

Composantes du signal :
  - bruit thermique gaussien complexe, d'écart-type SIGMA (en unités du bit de poids faible) ;
  - fouillis de sol (clutter) : échos forts aux courtes distances, quasi identiques d'une impulsion à l'autre ;
  - cibles ponctuelles : écho à une distance fixe, phase qui tourne d'une impulsion à l'autre (Doppler) ;
  - quantification sur 14 bits signés avec saturation.

Scénarios écrits dans data/ :
  noise_s4, noise_s16, noise_s64 : bruit seul (le pire cas pour la compression)
  scene_s16                      : bruit + fouillis + cibles (plus représentatif)

Exécuter : python gen_radar.py [taille_en_Mio]
"""

import json
import os
import sys

import numpy as np

N_RANGE = 4096           # cases distance par impulsion
ADC_BITS = 14
ADC_MAX = 2 ** (ADC_BITS - 1) - 1


def quantize(z):
    """Complexe -> int16 entrelacé, arrondi et saturé sur 14 bits."""
    iq = np.empty(z.shape + (2,), dtype=np.float64)
    iq[..., 0], iq[..., 1] = z.real, z.imag
    return np.clip(np.rint(iq), -ADC_MAX - 1, ADC_MAX).astype(np.int16)


def make_clutter(rng):
    """Profil de fouillis en distance : fort près du radar, décroissant, avec des points brillants."""
    r = np.arange(N_RANGE)
    amp = 3000.0 * np.exp(-r / 600.0) + 40.0
    amp *= rng.lognormal(0.0, 0.6, N_RANGE)
    phase = rng.uniform(0, 2 * np.pi, N_RANGE)
    return amp * np.exp(1j * phase)


def generate(path, n_pulses, sigma, scene, seed):
    rng = np.random.default_rng(seed)
    clutter = make_clutter(rng) if scene else None
    # cibles : (case distance, amplitude, fréquence Doppler en tours par impulsion)
    targets = [(900, 800.0, 0.013), (2100, 250.0, -0.071), (3300, 60.0, 0.24)] if scene else []
    block = 256  # impulsions générées à la fois, pour limiter la mémoire
    with open(path, "wb") as f:
        for p0 in range(0, n_pulses, block):
            n = min(block, n_pulses - p0)
            z = sigma * (rng.standard_normal((n, N_RANGE)) + 1j * rng.standard_normal((n, N_RANGE))) / np.sqrt(2)
            if scene:
                # fouillis légèrement fluctuant d'une impulsion à l'autre (vent, végétation)
                z += clutter[None, :] * (1 + 0.02 * rng.standard_normal((n, 1)))
                pulses = np.arange(p0, p0 + n)
                for cell, a, fd in targets:
                    z[:, cell] += a * np.exp(2j * np.pi * fd * pulses)
            quantize(z).tofile(f)


def main():
    size_mib = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    bytes_per_pulse = N_RANGE * 2 * 2  # I et Q sur 2 octets
    n_pulses = size_mib * 2**20 // bytes_per_pulse
    os.makedirs("data", exist_ok=True)
    scenarios = {"noise_s4": (4, False), "noise_s16": (16, False), "noise_s64": (64, False),
                 "scene_s16": (16, True)}
    for i, (name, (sigma, scene)) in enumerate(scenarios.items()):
        path = f"data/{name}.bin"
        generate(path, n_pulses, sigma, scene, seed=i)
        meta = dict(n_pulses=n_pulses, n_range=N_RANGE, adc_bits=ADC_BITS, sigma=sigma, scene=scene,
                    layout="int16 I/Q entrelacés, [impulsion][case distance][I,Q]")
        json.dump(meta, open(f"data/{name}.json", "w"), indent=1)
        print(f"{path} : {n_pulses} impulsions x {N_RANGE} cases, {os.path.getsize(path) / 2**20:.0f} Mio")


if __name__ == "__main__":
    main()
