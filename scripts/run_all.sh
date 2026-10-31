#!/bin/bash
# Campagne de mesures complète : tous les jeux de données, 1 à 8 threads, puis étude de la taille des blocs.
# Usage : ./scripts/run_all.sh   (après python gen_radar.py et la compilation dans build/)
set -euo pipefail

mkdir -p results
N_RANGE=4096

# Description de la machine, pour pouvoir interpréter les chiffres
{
  echo "date: $(date -Iseconds)"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  lscpu | grep -E "^(L1d|L2|L3)" | sed 's/  */ /g'
  echo "compilateur: $(cc --version | head -1)"
  echo "options: -O3 -march=native"
  for lib in liblz4-dev libzstd-dev libblosc2-dev zlib1g-dev; do
    echo "$lib: $(dpkg-query -W -f="\${Version}" "$lib" 2>/dev/null || echo inconnu)"
  done
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo aucun)"
} > results/machine.txt

HEADER="dataset,pipeline,threads,block_kib,ratio,comp_GBs,decomp_GBs"

echo "$HEADER" > results/bench.csv
for f in data/noise_s4.bin data/noise_s16.bin data/noise_s64.bin data/scene_s16.bin; do
  ./build/bench "$f" $N_RANGE 1,2,4,8 1024 results/bench.csv
done

echo "$HEADER" > results/blocks.csv
for kib in 64 256 1024 4096; do
  ./build/bench data/scene_s16.bin $N_RANGE 4 "$kib" results/blocks.csv
done
