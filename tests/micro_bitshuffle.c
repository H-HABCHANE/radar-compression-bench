// micro_bitshuffle.c — micro-benchmark du bitshuffle seul, pour l'analyse avec perf :
//   perf stat -e cycles,instructions,L1-dcache-load-misses,L1-dcache-store-misses ./micro_bitshuffle
// Usage : micro_bitshuffle [valeurs_par_bloc=524288] [scalaire=0]
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "filters.h"

int main(int argc, char **argv) {
    size_t n = argc > 1 ? (size_t)atol(argv[1]) : 524288;  // 1 Mio d'int16
    int scalar = argc > 2 && atoi(argv[2]);
    size_t total = 256u << 20, reps = total / (2 * n);     // 256 Mio traités au total
    uint16_t *src = malloc(2 * n);
    uint8_t *dst = malloc(2 * n);
    for (size_t i = 0; i < n; i++) src[i] = (uint16_t)(rand() & 0x3FFF);

    double t0 = omp_get_wtime();
    for (size_t r = 0; r < reps; r++)
        (scalar ? bitshuffle16_scalar : bitshuffle16)(src, dst, n);
    double t = omp_get_wtime() - t0;
    printf("%s, %zu valeurs par bloc : %.2f Go/s (contrôle %u)\n", scalar ? "scalaire" : "AVX2", n,
           (double)total / t / 1e9, dst[n / 3]);
    return 0;
}
