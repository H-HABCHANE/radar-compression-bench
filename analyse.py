"""Graphiques et tableau de synthèse à partir des mesures (results/*.csv).

Exécuter : python analyse.py   (après entropy.py et scripts/run_all.sh)
Sorties : figures/*.png
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette catégorielle de référence (3 premiers emplacements, validés pour le daltonisme)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "figure.facecolor": "white",
})
os.makedirs("figures", exist_ok=True)

NAMES = {"noise_s4": "bruit faible (σ = 2,8)", "noise_s16": "bruit moyen (σ = 11,3)",
         "noise_s64": "bruit fort (σ = 45)", "scene_s16": "scène réaliste"}
BEST = "delta + zigzag + bitshuffle + zstd-1"


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["dataset"] = os.path.basename(r["dataset"]).replace(".bin", "")
        for k in ("ratio", "comp_GBs", "decomp_GBs"):
            r[k] = float(r[k])
        r["threads"] = int(r["threads"])
    return rows


def family(name):
    if name.startswith("blosc2"):
        return "Blosc2 (référence open source)"
    if "+" in name and not name.startswith("["):
        return "pré-traitements de ce banc + compresseur"
    return "compresseur seul"


bench = load("results/bench.csv")
blocks = load("results/blocks.csv")
entropy = {r["dataset"]: r for r in csv.DictReader(open("results/entropy.csv"))}

# ---------- 1. Compromis taux / débit, scène réaliste, 4 threads ----------
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"compresseur seul": INK2, "pré-traitements de ce banc + compresseur": BLUE,
          "Blosc2 (référence open source)": ORANGE}
pts = [r for r in bench if r["dataset"] == "scene_s16" and r["threads"] == 4
       and not r["pipeline"].startswith(("[seul]", "copie")) and "scalaire" not in r["pipeline"]]
for fam, col in colors.items():
    sel = [r for r in pts if family(r["pipeline"]) == fam]
    ax.scatter([r["comp_GBs"] for r in sel], [r["ratio"] for r in sel], s=60, color=col, label=fam,
               edgecolor="white", linewidth=1.5, zorder=3)
for r in pts:  # étiquettes directes sur les chaînes qui comptent
    if r["pipeline"].startswith("delta") or r["pipeline"] in ("lz4", "zstd-1", "blosc2 bitshuffle + lz4"):
        best = r["pipeline"] == BEST  # à gauche du point : sinon il chevauche ses voisins
        ax.annotate(r["pipeline"], (r["comp_GBs"], r["ratio"]), xytext=(-8, -2) if best else (6, 4),
                    textcoords="offset points", fontsize=8, color=INK2, ha="right" if best else "left",
                    va="top" if best else "baseline")
bound = float(entropy["scene_s16"]["taux_max_delta"])
ax.axhline(bound, color=INK2, ls="--", lw=1)
ax.text(0.02, bound + 0.03, f"limite d'entropie après delta : {bound:.2f}", fontsize=8, color=INK2,
        transform=ax.get_yaxis_transform())
ax.axvline(1.0, color=INK2, ls=":", lw=1)
ax.text(1.03, 0.98, "objectif 1 Go/s", fontsize=8, color=INK2, transform=ax.get_xaxis_transform(), va="top")
ax.set(xlabel="débit de compression (Go/s, 4 cœurs)", ylabel="taux de compression",
       title="Scène radar réaliste : taux et débit de chaque chaîne", ylim=(0.9, bound + 0.25))
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/compromis_scene.png", dpi=150)

# ---------- 2. Taux atteint vs limite théorique, par jeu de données ----------
fig, ax = plt.subplots(figsize=(8, 3.8))
datasets = ["noise_s4", "noise_s16", "noise_s64", "scene_s16"]
y = range(len(datasets))
best_ratio = []
for d in datasets:
    rs = [r for r in bench if r["dataset"] == d and r["threads"] == 4 and "scalaire" not in r["pipeline"]]
    best_ratio.append(max(rs, key=lambda r: r["ratio"]))
bound0 = [min(float(entropy[d]["taux_max"]), 99) for d in datasets]
bound_d = [float(entropy[d]["taux_max_delta"]) for d in datasets]
limit = [max(a, b) for a, b in zip(bound0, bound_d)]
ax.barh([i - 0.2 for i in y], [r["ratio"] for r in best_ratio], height=0.36, color=BLUE,
        label="meilleur taux mesuré")
ax.barh([i + 0.2 for i in y], limit, height=0.36, color=AQUA, label="limite estimée (entropie)")
for i, (r, lim) in enumerate(zip(best_ratio, limit)):
    ax.text(r["ratio"] + 0.03, i - 0.2, f"{r['ratio']:.2f}  ({r['pipeline']})", va="center", fontsize=8, color=INK)
    ax.text(lim + 0.03, i + 0.2, f"{lim:.2f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(list(y), [NAMES[d] for d in datasets])
ax.invert_yaxis()
ax.set(xlabel="taux de compression", title="Taux mesuré et limite théorique", xlim=(0, max(limit) * 1.75))
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/taux_vs_limite.png", dpi=150)

# ---------- 3. Passage à l'échelle, scène réaliste ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
chains = [(BEST, BLUE), ("delta + zigzag + bitshuffle + lz4", ORANGE), ("blosc2 bitshuffle + lz4", AQUA)]
for ax, key, title in [(axes[0], "comp_GBs", "Compression"), (axes[1], "decomp_GBs", "Décompression")]:
    for name, col in chains:
        rs = sorted([r for r in bench if r["dataset"] == "scene_s16" and r["pipeline"] == name],
                    key=lambda r: r["threads"])
        ax.plot([r["threads"] for r in rs], [r[key] for r in rs], "o-", color=col, ms=6, label=name)
        dy = {BEST: -5, "delta + zigzag + bitshuffle + lz4": 5}.get(name, 0)  # valeurs finales proches
        ax.annotate(f"{rs[-1][key]:.2f}", (rs[-1]["threads"], rs[-1][key]), xytext=(5, dy),
                    textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.axhline(1.0, color=INK2, ls=":", lw=1)
    ax.set(xlabel="threads (4 cœurs physiques)", title=title, xticks=[1, 2, 4, 8])
axes[0].set_ylabel("débit (Go/s)")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figures/scaling_scene.png", dpi=150)

# ---------- 4. Taille des blocs : deux mesures, deux graphiques (pas de double axe) ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for name, col in chains[:2]:
    rs = sorted([r for r in blocks if r["pipeline"] == name], key=lambda r: int(r["block_kib"]))
    kib = [int(r["block_kib"]) for r in rs]
    axes[0].plot(kib, [r["ratio"] for r in rs], "o-", color=col, ms=6, label=name)
    axes[1].plot(kib, [r["comp_GBs"] for r in rs], "o-", color=col, ms=6, label=name)
for ax, ylabel in zip(axes, ["taux de compression", "débit de compression (Go/s, 4 cœurs)"]):
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256, 1024, 4096], ["64 Kio", "256 Kio", "1 Mio", "4 Mio"])
    ax.set(xlabel="taille des blocs", ylabel=ylabel)
axes[0].legend(fontsize=8)
fig.suptitle("Scène réaliste : influence de la taille des blocs", fontsize=10)
fig.tight_layout()
fig.savefig("figures/taille_blocs.png", dpi=150)

# ---------- 5. Étapes d'optimisation ----------
steps = list(csv.DictReader(open("results/optim_steps.csv")))
fig, ax = plt.subplots(figsize=(9, 3.2))
yy = range(len(steps))
vals = [float(s["comp_GBs_4threads"]) for s in steps]
ax.barh(list(yy), vals, height=0.55, color=BLUE)
for i, v in enumerate(vals):
    ax.text(v + 0.03, i, f"{v:.2f} Go/s", va="center", fontsize=8, color=INK)
ax.axvline(1.0, color=INK2, ls=":", lw=1)
ax.set_yticks(list(yy), [s["etape"] for s in steps])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set(xlabel="débit de compression (Go/s, 4 cœurs)", xlim=(0, max(vals) * 1.25),
       title="Étapes d'optimisation de la chaîne delta + zigzag + bitshuffle + zstd-1")
fig.tight_layout()
fig.savefig("figures/etapes_optimisation.png", dpi=150)

# ---------- Tableau de synthèse ----------
print(f"{'données':24s} {'chaîne':44s} {'taux':>6s} {'comp.':>7s} {'décomp.':>8s}   (4 threads)")
for d in datasets:
    for name in ["lz4", "zstd-1", "blosc2 bitshuffle + lz4", "delta + zigzag + bitshuffle + lz4",
                 "zigzag + bitshuffle + zstd-1", BEST, "delta + zigzag + bitshuffle + zstd--1"]:
        r = next(r for r in bench if r["dataset"] == d and r["pipeline"] == name and r["threads"] == 4)
        print(f"{NAMES[d]:24s} {name:44s} {r['ratio']:6.3f} {r['comp_GBs']:7.2f} {r['decomp_GBs']:8.2f}")
    print()
