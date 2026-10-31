// test_filters.c — vérifie que chaque pré-traitement est exactement réversible,
// et compare le bitshuffle rapide (transposition 8 x 8) à une version bit par bit évidente.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "filters.h"

static int failures = 0;

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            fprintf(stderr, "ÉCHEC : " __VA_ARGS__); \
            fprintf(stderr, "\n");               \
            failures++;                          \
        }                                        \
    } while (0)

// Référence : bit p de la valeur i -> bit (i % 8) de l'octet p * (n / 8) + i / 8
static void bitshuffle16_naive(const uint16_t *src, uint8_t *dst, size_t n) {
    size_t nb = n / 8;
    memset(dst, 0, 2 * n);
    for (size_t i = 0; i < 8 * nb; i++)
        for (int p = 0; p < 16; p++)
            if (src[i] >> p & 1) dst[p * nb + i / 8] |= (uint8_t)(1u << (i % 8));
    memcpy(dst + 16 * nb, src + 8 * nb, (n - 8 * nb) * sizeof(uint16_t));
}

static void fill(int16_t *x, size_t n, int kind) {
    for (size_t i = 0; i < n; i++) {
        switch (kind) {
            case 0: x[i] = (int16_t)(rand() & 0xFFFF); break;             // valeurs quelconques
            case 1: x[i] = (int16_t)(rand() % 41 - 20); break;            // petites valeurs signées
            default: x[i] = (int16_t)(i % 2 ? -32768 : 32767); break;     // extrêmes (dépassements du delta)
        }
    }
}

int main(void) {
    srand(12345);
    const size_t sizes[] = {0, 1, 7, 8, 9, 31, 32, 33, 64, 511, 512, 513, 1000, 8192 * 3 + 5, 524288};
    const size_t strides[] = {1, 8, 32, 8192};
    const int flag_sets[] = {0, F_DELTA, F_ZIGZAG, F_SHUFFLE, F_BITSHUFFLE, F_DELTA | F_ZIGZAG,
                             F_ZIGZAG | F_BITSHUFFLE, F_DELTA | F_SHUFFLE, F_DELTA | F_BITSHUFFLE,
                             F_DELTA | F_ZIGZAG | F_SHUFFLE, F_DELTA | F_ZIGZAG | F_BITSHUFFLE,
                             F_DELTA | F_ZIGZAG | F_BITSHUFFLE | F_SCALAR};
    int tests = 0;

    for (size_t s = 0; s < sizeof sizes / sizeof *sizes; s++) {
        size_t n = sizes[s];
        int16_t *x = malloc(n * 2 + 2), *y = malloc(n * 2 + 2);
        uint8_t *a = malloc(n * 2 + 2), *b = malloc(n * 2 + 2), *tmp = malloc(n * 2 + 2);

        for (int kind = 0; kind < 3; kind++) {
            fill(x, n, kind);

            bitshuffle16_naive((uint16_t *)x, b, n);
            bitshuffle16((uint16_t *)x, a, n);
            CHECK(memcmp(a, b, n * 2) == 0, "bitshuffle (AVX2) différent de la référence (n = %zu)", n);
            bitshuffle16_scalar((uint16_t *)x, a, n);
            CHECK(memcmp(a, b, n * 2) == 0, "bitshuffle scalaire différent de la référence (n = %zu)", n);
            bitunshuffle16(b, (uint16_t *)y, n);
            CHECK(memcmp(x, y, n * 2) == 0, "bitunshuffle (AVX2) faux (n = %zu)", n);
            bitunshuffle16_scalar(b, (uint16_t *)y, n);
            CHECK(memcmp(x, y, n * 2) == 0, "bitunshuffle scalaire faux (n = %zu)", n);
            tests += 4;

            for (size_t f = 0; f < sizeof flag_sets / sizeof *flag_sets; f++)
                for (size_t st = 0; st < sizeof strides / sizeof *strides; st++) {
                    const void *out = filters_forward(flag_sets[f], x, tmp, a, n, strides[st]);
                    memcpy(b, out, n * 2);  // copie : le résultat peut être dans tmp
                    filters_inverse(flag_sets[f], b, tmp, y, n, strides[st]);
                    CHECK(memcmp(x, y, n * 2) == 0, "aller-retour faux : flags = %d, n = %zu, stride = %zu",
                          flag_sets[f], n, strides[st]);
                    tests++;
                }
        }
        free(x), free(y), free(a), free(b), free(tmp);
    }

    // Zigzag : petites valeurs signées -> petites valeurs non signées
    int16_t v[5] = {0, -1, 1, -2, 2};
    uint16_t z[5];
    zigzag_forward(v, z, 5);
    CHECK(z[0] == 0 && z[1] == 1 && z[2] == 2 && z[3] == 3 && z[4] == 4, "zigzag : mauvaises valeurs");
    tests++;

    printf("%d tests, %d échec(s)\n", tests, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
