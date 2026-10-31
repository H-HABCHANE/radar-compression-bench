// bench.c — banc de test de compression sans perte de données radar I/Q (int16).
//
// Les données sont découpées en blocs indépendants (par défaut 1 Mio = 64 impulsions), compressés
// en parallèle avec OpenMP. Pour chaque chaîne « pré-traitements + compresseur » on mesure :
//   - le taux de compression (taille d'origine / taille compressée) ;
//   - le débit de compression et de décompression, en Go/s de données d'origine ;
// et on vérifie octet par octet que la décompression redonne exactement les données d'origine.
//
// Usage : bench <fichier.bin> <cases_distance> [threads=1,2,4,8] [taille_bloc_Kio=1024] [csv]
#include <blosc2.h>
#include <lz4.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include <zstd.h>

#include "filters.h"

typedef enum { C_NONE, C_LZ4, C_ZSTD, C_ZLIB, C_BLOSC2 } Codec;

typedef struct {
    const char *name;
    int flags;   // pré-traitements faits par ce banc (filters.h)
    Codec codec;
    int level;
    int blosc_filter;  // pour C_BLOSC2 : pré-traitement fait par Blosc2 (implémentation SIMD)
    int blosc_codec;
} Pipeline;

static const Pipeline PIPELINES[] = {
    {"copie (memcpy)", 0, C_NONE, 0, 0, 0},
    {"zlib-1", 0, C_ZLIB, 1, 0, 0},
    {"lz4", 0, C_LZ4, 0, 0, 0},
    {"zstd-1", 0, C_ZSTD, 1, 0, 0},
    {"zstd-3", 0, C_ZSTD, 3, 0, 0},
    // Pré-traitements seuls, sans compresseur : leur coût propre
    {"[seul] bitshuffle scalaire", F_BITSHUFFLE | F_SCALAR, C_NONE, 0, 0, 0},
    {"[seul] bitshuffle AVX2", F_BITSHUFFLE, C_NONE, 0, 0, 0},
    {"[seul] delta + zigzag + bitshuffle", F_DELTA | F_ZIGZAG | F_BITSHUFFLE, C_NONE, 0, 0, 0},
    {"shuffle + lz4", F_SHUFFLE, C_LZ4, 0, 0, 0},
    {"shuffle + zstd-1", F_SHUFFLE, C_ZSTD, 1, 0, 0},
    {"bitshuffle + lz4", F_BITSHUFFLE, C_LZ4, 0, 0, 0},
    {"bitshuffle + zstd-1", F_BITSHUFFLE, C_ZSTD, 1, 0, 0},
    {"zigzag + bitshuffle + lz4", F_ZIGZAG | F_BITSHUFFLE, C_LZ4, 0, 0, 0},
    {"zigzag + bitshuffle + zstd-1", F_ZIGZAG | F_BITSHUFFLE, C_ZSTD, 1, 0, 0},
    {"delta + zigzag + bitshuffle + lz4", F_DELTA | F_ZIGZAG | F_BITSHUFFLE, C_LZ4, 0, 0, 0},
    {"delta + zigzag + bitshuffle scalaire + zstd-1", F_DELTA | F_ZIGZAG | F_BITSHUFFLE | F_SCALAR, C_ZSTD, 1, 0, 0},
    {"delta + zigzag + bitshuffle + zstd-1", F_DELTA | F_ZIGZAG | F_BITSHUFFLE, C_ZSTD, 1, 0, 0},
    {"delta + zigzag + bitshuffle + zstd--1", F_DELTA | F_ZIGZAG | F_BITSHUFFLE, C_ZSTD, -1, 0, 0},
    {"blosc2 shuffle + lz4", 0, C_BLOSC2, 5, BLOSC_SHUFFLE, BLOSC_LZ4},
    {"blosc2 bitshuffle + lz4", 0, C_BLOSC2, 5, BLOSC_BITSHUFFLE, BLOSC_LZ4},
    {"blosc2 bitshuffle + zstd-1", 0, C_BLOSC2, 1, BLOSC_BITSHUFFLE, BLOSC_ZSTD},
};
#define N_PIPELINES (sizeof PIPELINES / sizeof *PIPELINES)

// Tampons et contextes propres à chaque thread
typedef struct {
    uint8_t *tmp1, *tmp2;
    ZSTD_CCtx *zc;
    ZSTD_DCtx *zd;
    blosc2_context *bc, *bd;
} Work;

static size_t bound(size_t n) {
    size_t b = (size_t)LZ4_compressBound((int)n);
    if (ZSTD_compressBound(n) > b) b = ZSTD_compressBound(n);
    if (compressBound(n) > b) b = compressBound(n);
    return b + BLOSC2_MAX_OVERHEAD;
}

static size_t compress_block(const Pipeline *p, Work *w, const int16_t *src, size_t bytes, uint8_t *dst,
                             size_t cap, size_t stride) {
    size_t n = bytes / 2;
    const void *in = filters_forward(p->flags, src, w->tmp1, w->tmp2, n, stride);
    switch (p->codec) {
        case C_NONE: memcpy(dst, in, bytes); return bytes;
        case C_LZ4: return (size_t)LZ4_compress_default(in, (char *)dst, (int)bytes, (int)cap);
        case C_ZSTD: return ZSTD_compressCCtx(w->zc, dst, cap, in, bytes, p->level);
        case C_ZLIB: {
            uLongf len = cap;
            return compress2(dst, &len, in, bytes, p->level) == Z_OK ? len : 0;
        }
        case C_BLOSC2: {
            int r = blosc2_compress_ctx(w->bc, in, (int32_t)bytes, dst, (int32_t)cap);
            return r > 0 ? (size_t)r : 0;
        }
    }
    return 0;
}

static int decompress_block(const Pipeline *p, Work *w, const uint8_t *src, size_t csize, int16_t *out,
                            size_t bytes, size_t stride) {
    size_t n = bytes / 2;
    uint8_t *raw = p->flags ? w->tmp2 : (uint8_t *)out;  // sans pré-traitement : directement dans out
    size_t got = 0;
    switch (p->codec) {
        case C_NONE: memcpy(raw, src, bytes); got = bytes; break;
        case C_LZ4: got = (size_t)LZ4_decompress_safe((const char *)src, (char *)raw, (int)csize, (int)bytes); break;
        case C_ZSTD: got = ZSTD_decompressDCtx(w->zd, raw, bytes, src, csize); break;
        case C_ZLIB: {
            uLongf len = bytes;
            got = uncompress(raw, &len, src, csize) == Z_OK ? len : 0;
            break;
        }
        case C_BLOSC2: got = (size_t)blosc2_decompress_ctx(w->bd, src, (int32_t)csize, raw, (int32_t)bytes); break;
    }
    if (got != bytes) return -1;
    if (p->flags) filters_inverse(p->flags, raw, w->tmp1, out, n, stride);
    return 0;
}

static void setup_blosc(Work *w, const Pipeline *p) {
    if (w->bc) blosc2_free_ctx(w->bc), w->bc = NULL;
    if (w->bd) blosc2_free_ctx(w->bd), w->bd = NULL;
    if (p->codec != C_BLOSC2) return;
    blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
    cp.typesize = 2;
    cp.compcode = (uint8_t)p->blosc_codec;
    cp.clevel = (uint8_t)p->level;
    cp.nthreads = 1;  // le parallélisme vient d'OpenMP, comme pour les autres chaînes
    for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
    cp.filters[BLOSC2_MAX_FILTERS - 1] = (uint8_t)p->blosc_filter;
    blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS;
    dp.nthreads = 1;
    w->bc = blosc2_create_cctx(cp);
    w->bd = blosc2_create_dctx(dp);
}

static int parse_list(const char *s, int *out, int max) {
    int n = 0;
    char *copy = strdup(s), *tok = strtok(copy, ",");
    while (tok && n < max) out[n++] = atoi(tok), tok = strtok(NULL, ",");
    free(copy);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage : %s fichier.bin cases_distance [threads=1,2,4,8] [bloc_Kio=1024] [csv]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *path = argv[1];
    size_t stride = 2 * (size_t)atol(argv[2]);  // une impulsion = n_range valeurs I + n_range valeurs Q
    int threads[16], n_threads = parse_list(argc > 3 ? argv[3] : "1,2,4,8", threads, 16);
    size_t block = (size_t)(argc > 4 ? atol(argv[4]) : 1024) * 1024;
    const char *csv = argc > 5 ? argv[5] : NULL;

    // 1. Lecture du fichier en mémoire : on mesure le compresseur, pas le disque
    FILE *f = fopen(path, "rb");
    if (!f) return perror(path), EXIT_FAILURE;
    fseek(f, 0, SEEK_END);
    size_t total = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    int16_t *data = malloc(total), *back = malloc(total);
    if (fread(data, 1, total, f) != total) return fprintf(stderr, "lecture incomplète\n"), EXIT_FAILURE;
    fclose(f);

    size_t n_blocks = (total + block - 1) / block, cap = bound(block);
    uint8_t *comp = malloc(n_blocks * cap);
    size_t *csize = malloc(n_blocks * sizeof(size_t));
    int max_threads = omp_get_num_procs();
    Work *work = calloc((size_t)max_threads, sizeof(Work));
    for (int t = 0; t < max_threads; t++) {
        work[t].tmp1 = malloc(block), work[t].tmp2 = malloc(block);
        work[t].zc = ZSTD_createCCtx(), work[t].zd = ZSTD_createDCtx();
    }
    blosc2_init();

    // Pages touchées une première fois : le premier test ne paie pas les défauts de page
    memset(back, 0, total);
    memset(comp, 0, n_blocks * cap);

    FILE *out = csv ? fopen(csv, "a") : NULL;
    printf("%s : %.0f Mio, blocs de %zu Kio, %zu blocs\n", path, total / 1048576.0, block / 1024, n_blocks);
    printf("%-46s %4s %8s %12s %12s\n", "chaîne", "thr", "taux", "comp. Go/s", "décomp. Go/s");

    for (size_t ip = 0; ip < N_PIPELINES; ip++) {
        const Pipeline *p = &PIPELINES[ip];
        for (int t = 0; t < max_threads; t++) setup_blosc(&work[t], p);

        for (int it = 0; it < n_threads; it++) {
            int nt = threads[it];
            if (nt > max_threads) continue;
            double best_c = 1e30, best_d = 1e30;
            size_t comp_total = 0;
            // Meilleur temps sur 3 essais ; un seul essai si le premier dépasse 3 s (compresseurs lents)
            for (int rep = 0; rep < 3; rep++) {
                double t0 = omp_get_wtime();
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
                for (size_t b = 0; b < n_blocks; b++) {
                    size_t bytes = (b + 1) * block > total ? total - b * block : block;
                    csize[b] = compress_block(p, &work[omp_get_thread_num()], data + b * block / 2, bytes,
                                              comp + b * cap, cap, stride);
                }
                double t1 = omp_get_wtime();
                int errors = 0;
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1) reduction(+ : errors)
                for (size_t b = 0; b < n_blocks; b++) {
                    size_t bytes = (b + 1) * block > total ? total - b * block : block;
                    errors += decompress_block(p, &work[omp_get_thread_num()], comp + b * cap, csize[b],
                                               back + b * block / 2, bytes, stride) != 0;
                }
                double t2 = omp_get_wtime();

                // 2. Vérification : la compression doit être sans perte, octet par octet
                if (errors || memcmp(data, back, total) != 0) {
                    fprintf(stderr, "ERREUR : %s ne redonne pas les données d'origine\n", p->name);
                    return EXIT_FAILURE;
                }
                memset(back, 0, total);  // le prochain essai ne peut pas réussir par hasard
                comp_total = 0;
                for (size_t b = 0; b < n_blocks; b++) comp_total += csize[b];
                if (t1 - t0 < best_c) best_c = t1 - t0;
                if (t2 - t1 < best_d) best_d = t2 - t1;
                if (t2 - t0 > 3.0) break;
            }
            double ratio = (double)total / comp_total;
            double gc = total / best_c / 1e9, gd = total / best_d / 1e9;
            printf("%-46s %4d %8.3f %12.3f %12.3f\n", p->name, nt, ratio, gc, gd);
            fflush(stdout);
            if (out)
                fprintf(out, "%s,%s,%d,%zu,%.5f,%.4f,%.4f\n", path, p->name, nt, block / 1024, ratio, gc, gd);
        }
    }
    if (out) fclose(out);
    for (int t = 0; t < max_threads; t++) setup_blosc(&work[t], &PIPELINES[0]);  // libère les contextes
    blosc2_destroy();
    return EXIT_SUCCESS;
}
