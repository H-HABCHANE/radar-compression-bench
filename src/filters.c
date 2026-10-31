// filters.c — pré-traitements réversibles : delta, zigzag, byte shuffle, bitshuffle.
#include "filters.h"

#include <string.h>

// Delta d'une impulsion à l'autre : le fouillis de sol, quasi identique entre deux impulsions,
// disparaît ; le bruit, lui, voit sa variance doubler. L'arithmétique non signée (modulo 2^16)
// garantit un aller-retour exact, même en cas de dépassement.
void delta_forward(const int16_t *src, int16_t *dst, size_t n, size_t stride) {
    size_t first = stride < n ? stride : n;
    memcpy(dst, src, first * sizeof(int16_t));  // la première impulsion est gardée telle quelle
    for (size_t i = first; i < n; i++)
        dst[i] = (int16_t)((uint16_t)src[i] - (uint16_t)src[i - stride]);
}

void delta_inverse(const int16_t *src, int16_t *dst, size_t n, size_t stride) {
    size_t first = stride < n ? stride : n;
    memcpy(dst, src, first * sizeof(int16_t));
    for (size_t i = first; i < n; i++)
        dst[i] = (int16_t)((uint16_t)src[i] + (uint16_t)dst[i - stride]);
}

// Zigzag : 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ... Un petit nombre négatif (0xFFFE en complément
// à deux) devient un petit nombre positif : ses bits de poids fort passent de 1 à 0.
void zigzag_forward(const int16_t *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = (uint16_t)(((uint16_t)src[i] << 1) ^ (uint16_t)(src[i] >> 15));
}

void zigzag_inverse(const uint16_t *src, int16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = (int16_t)((src[i] >> 1) ^ (uint16_t)(-(src[i] & 1)));
}

// Byte shuffle (valeurs de 2 octets) : [lo0 hi0 lo1 hi1 ...] -> [lo0 lo1 ... | hi0 hi1 ...]
void shuffle2(const uint8_t *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[2 * i];
        dst[n + i] = src[2 * i + 1];
    }
}

void unshuffle2(const uint8_t *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[2 * i] = src[i];
        dst[2 * i + 1] = src[n + i];
    }
}

// Transposition d'une matrice 8 x 8 bits rangée dans un uint64 (octet r = ligne r, bit c = colonne c) :
// le bit c de l'octet r devient le bit r de l'octet c. Elle est sa propre inverse.
// Méthode par échanges de blocs (Hacker's Delight, section 7-3).
static inline uint64_t transpose8(uint64_t x) {
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    x = x ^ t ^ (t << 28);
    return x;
}

// Bitshuffle (valeurs de 16 bits) : le plan p (p = 0..15) contient le bit p de chaque valeur,
// 8 valeurs par octet. Avec un convertisseur 14 bits et du bruit faible, les plans de poids fort
// ne contiennent presque que des 0 (ou que le bit de signe) : ils se compressent très bien.
// Par paquets de 8 valeurs : on transpose la matrice 8 x 8 de leurs octets faibles, puis forts.
// Les valeurs au-delà d'un multiple de 8 sont recopiées telles quelles à la fin.
// Paquets de 8 valeurs numéros m0 à m1 - 1 (nb paquets en tout)
static void bitshuffle16_range(const uint16_t *src, uint8_t *dst, size_t nb, size_t m0, size_t m1) {
    for (size_t m = m0; m < m1; m++) {
        const uint16_t *v = src + 8 * m;
        for (int k = 0; k < 2; k++) {  // k = 0 : octets faibles, k = 1 : octets forts
            uint64_t x = 0;
            for (int j = 0; j < 8; j++)
                x |= (uint64_t)((v[j] >> (8 * k)) & 0xFF) << (8 * j);
            x = transpose8(x);  // octet b de x = bit b des 8 valeurs
            for (int b = 0; b < 8; b++)
                dst[(size_t)(8 * k + b) * nb + m] = (uint8_t)(x >> (8 * b));
        }
    }
}

void bitshuffle16_scalar(const uint16_t *src, uint8_t *dst, size_t n) {
    size_t nb = n / 8;
    bitshuffle16_range(src, dst, nb, 0, nb);
    memcpy(dst + 16 * nb, src + 8 * nb, (n - 8 * nb) * sizeof(uint16_t));
}

#ifdef __AVX2__
#include <immintrin.h>

// Version AVX2, 32 valeurs à la fois :
//   1. on sépare les 32 octets faibles et les 32 octets forts dans deux registres de 256 bits ;
//   2. _mm256_movemask_epi8 rassemble le bit de poids fort des 32 octets en un entier de 32 bits :
//      c'est directement 4 octets d'un plan de bits ;
//   3. un décalage à gauche amène le bit suivant en position de poids fort, 8 fois par registre.
// 16 movemask remplacent 32 x 16 extractions de bits.
//
// Les 16 plans sont espacés de n / 8 octets, souvent une puissance de 2 : ils tombent alors dans le
// même ensemble du cache L1 (8 voies) et chaque écriture de 4 octets rate le cache (mesuré avec perf).
// On accumule donc 64 octets par plan (une ligne de cache, soit 16 paquets de 32 valeurs) dans un
// tampon local, puis on écrit chaque ligne d'un coup.
static inline void bitshuffle32_avx2(const uint16_t *src, uint32_t masks[16]) {
    // Dans chaque moitié de 128 bits : 8 octets faibles puis 8 octets forts
    const __m256i split = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                           0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m256i a = _mm256_loadu_si256((const __m256i *)src);         // valeurs 0 à 15
    __m256i b = _mm256_loadu_si256((const __m256i *)(src + 16));  // valeurs 16 à 31
    // a = [faibles 0-7 | forts 0-7 | faibles 8-15 | forts 8-15] -> [faibles 0-15 | forts 0-15]
    a = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(a, split), 0xD8);
    b = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(b, split), 0xD8);
    __m256i lo = _mm256_permute2x128_si256(a, b, 0x20);  // octets faibles des 32 valeurs
    __m256i hi = _mm256_permute2x128_si256(a, b, 0x31);  // octets forts des 32 valeurs
    for (int bit = 7; bit >= 0; bit--) {
        // décalage de (7 - bit) sur des mots de 16 bits : chaque octet garde ses propres bits
        masks[bit] = (uint32_t)_mm256_movemask_epi8(_mm256_slli_epi16(lo, 7 - bit));
        masks[8 + bit] = (uint32_t)_mm256_movemask_epi8(_mm256_slli_epi16(hi, 7 - bit));
    }
}

void bitshuffle16(const uint16_t *src, uint8_t *dst, size_t n) {
    size_t nb = n / 8, n32 = n / 32, g = 0;
    uint32_t line[16][16];  // [plan][paquet] : 64 octets par plan
    for (; g + 16 <= n32; g += 16) {
        uint32_t masks[16];
        for (int k = 0; k < 16; k++) {
            bitshuffle32_avx2(src + 32 * (g + k), masks);
            for (int p = 0; p < 16; p++) line[p][k] = masks[p];
        }
        for (int p = 0; p < 16; p++) memcpy(dst + (size_t)p * nb + 4 * g, line[p], 64);
    }
    for (; g < n32; g++) {  // paquets de 32 restants
        uint32_t masks[16];
        bitshuffle32_avx2(src + 32 * g, masks);
        for (int p = 0; p < 16; p++) memcpy(dst + (size_t)p * nb + 4 * g, &masks[p], 4);
    }
    bitshuffle16_range(src, dst, nb, 4 * n32, nb);  // derniers paquets de 8
    memcpy(dst + 16 * nb, src + 8 * nb, (n - 8 * nb) * sizeof(uint16_t));
}

// Delta + zigzag de 32 valeurs à partir de l'indice i, calculés dans les registres
static inline void delta_zigzag32_avx2(const int16_t *src, size_t i, size_t stride, uint16_t out[32]) {
    for (int h = 0; h < 32; h += 16) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(src + i + h));
        if (i >= stride)  // première impulsion du bloc : pas d'impulsion précédente, zigzag seul
            x = _mm256_sub_epi16(x, _mm256_loadu_si256((const __m256i *)(src + i + h - stride)));
        x = _mm256_xor_si256(_mm256_slli_epi16(x, 1), _mm256_srai_epi16(x, 15));
        _mm256_storeu_si256((__m256i *)(out + h), x);
    }
}

// Delta, zigzag et bitshuffle en un seul passage sur la mémoire : on lit l'impulsion courante
// et la précédente, et on n'écrit que les plans de bits (au lieu de trois lectures et deux écritures).
void delta_zigzag_bitshuffle16(const int16_t *src, uint8_t *dst, uint16_t *tmp, size_t n, size_t stride) {
    if (n % 32 != 0 || stride % 32 != 0) {  // cas général : deux passages
        delta_zigzag_forward(src, tmp, n, stride);
        bitshuffle16(tmp, dst, n);
        return;
    }
    size_t nb = n / 8, n32 = n / 32, g = 0;
    uint32_t line[16][16];
    uint16_t v[32];
    for (; g + 16 <= n32; g += 16) {
        uint32_t masks[16];
        for (int k = 0; k < 16; k++) {
            delta_zigzag32_avx2(src, 32 * (g + k), stride, v);
            bitshuffle32_avx2(v, masks);
            for (int p = 0; p < 16; p++) line[p][k] = masks[p];
        }
        for (int p = 0; p < 16; p++) memcpy(dst + (size_t)p * nb + 4 * g, line[p], 64);
    }
    for (; g < n32; g++) {
        uint32_t masks[16];
        delta_zigzag32_avx2(src, 32 * g, stride, v);
        bitshuffle32_avx2(v, masks);
        for (int p = 0; p < 16; p++) memcpy(dst + (size_t)p * nb + 4 * g, &masks[p], 4);
    }
}

// Inverse de bitshuffle32_avx2 : à partir des 16 masques de 32 bits, reconstruit 32 valeurs.
// Pour chaque plan : l'octet j du registre reçoit l'octet du masque qui contient le bit j,
// on isole ce bit, et cmpeq le transforme en 0x00 ou 0xFF, qu'on réduit au bit du plan.
static inline void bitunshuffle32_avx2(const uint32_t masks[16], uint16_t *dst) {
    const __m256i sel = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bitpat = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
    for (int bit = 0; bit < 8; bit++) {
        __m256i w = _mm256_set1_epi8((char)(1 << bit));
        __m256i x = _mm256_shuffle_epi8(_mm256_set1_epi32((int)masks[bit]), sel);
        lo = _mm256_or_si256(lo, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(x, bitpat), bitpat), w));
        x = _mm256_shuffle_epi8(_mm256_set1_epi32((int)masks[8 + bit]), sel);
        hi = _mm256_or_si256(hi, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(x, bitpat), bitpat), w));
    }
    // Entrelace octets faibles et forts : unpack travaille par moitié de 128 bits
    __m256i u0 = _mm256_unpacklo_epi8(lo, hi);  // valeurs 0-7 | 16-23
    __m256i u1 = _mm256_unpackhi_epi8(lo, hi);  // valeurs 8-15 | 24-31
    _mm256_storeu_si256((__m256i *)dst, _mm256_permute2x128_si256(u0, u1, 0x20));
    _mm256_storeu_si256((__m256i *)(dst + 16), _mm256_permute2x128_si256(u0, u1, 0x31));
}

static void bitunshuffle16_range(const uint8_t *src, uint16_t *dst, size_t nb, size_t m0, size_t m1);

void bitunshuffle16(const uint8_t *src, uint16_t *dst, size_t n) {
    size_t nb = n / 8, n32 = n / 32, g = 0;
    uint32_t line[16][16];  // même tampon par ligne de cache qu'à la compression, en lecture
    for (; g + 16 <= n32; g += 16) {
        for (int p = 0; p < 16; p++) memcpy(line[p], src + (size_t)p * nb + 4 * g, 64);
        for (int k = 0; k < 16; k++) {
            uint32_t masks[16];
            for (int p = 0; p < 16; p++) masks[p] = line[p][k];
            bitunshuffle32_avx2(masks, dst + 32 * (g + k));
        }
    }
    for (; g < n32; g++) {
        uint32_t masks[16];
        for (int p = 0; p < 16; p++) memcpy(&masks[p], src + (size_t)p * nb + 4 * g, 4);
        bitunshuffle32_avx2(masks, dst + 32 * g);
    }
    bitunshuffle16_range(src, dst, nb, 4 * n32, nb);
    memcpy(dst + 8 * nb, src + 16 * nb, (n - 8 * nb) * sizeof(uint16_t));
}
#else
void bitshuffle16(const uint16_t *src, uint8_t *dst, size_t n) { bitshuffle16_scalar(src, dst, n); }
void bitunshuffle16(const uint8_t *src, uint16_t *dst, size_t n) { bitunshuffle16_scalar(src, dst, n); }
void delta_zigzag_bitshuffle16(const int16_t *src, uint8_t *dst, uint16_t *tmp, size_t n, size_t stride) {
    delta_zigzag_forward(src, tmp, n, stride);
    bitshuffle16(tmp, dst, n);
}
#endif

static void bitunshuffle16_range(const uint8_t *src, uint16_t *dst, size_t nb, size_t m0, size_t m1) {
    for (size_t m = m0; m < m1; m++) {
        uint16_t *v = dst + 8 * m;
        for (int j = 0; j < 8; j++) v[j] = 0;
        for (int k = 0; k < 2; k++) {
            uint64_t x = 0;
            for (int b = 0; b < 8; b++)
                x |= (uint64_t)src[(size_t)(8 * k + b) * nb + m] << (8 * b);
            x = transpose8(x);
            for (int j = 0; j < 8; j++)
                v[j] |= (uint16_t)(((x >> (8 * j)) & 0xFF) << (8 * k));
        }
    }
}

void bitunshuffle16_scalar(const uint8_t *src, uint16_t *dst, size_t n) {
    size_t nb = n / 8;
    bitunshuffle16_range(src, dst, nb, 0, nb);
    memcpy(dst + 8 * nb, src + 16 * nb, (n - 8 * nb) * sizeof(uint16_t));
}

// Delta puis zigzag en une seule passe sur la mémoire (au lieu de deux)
void delta_zigzag_forward(const int16_t *src, uint16_t *dst, size_t n, size_t stride) {
    size_t first = stride < n ? stride : n;
    zigzag_forward(src, dst, first);
    for (size_t i = first; i < n; i++) {
        uint16_t d = (uint16_t)((uint16_t)src[i] - (uint16_t)src[i - stride]);
        dst[i] = (uint16_t)((d << 1) ^ (uint16_t)((int16_t)d >> 15));
    }
}

void delta_zigzag_inverse(const uint16_t *src, int16_t *dst, size_t n, size_t stride) {
    size_t first = stride < n ? stride : n;
    zigzag_inverse(src, dst, first);
    for (size_t i = first; i < n; i++) {
        uint16_t d = (uint16_t)((src[i] >> 1) ^ (uint16_t)(-(src[i] & 1)));
        dst[i] = (int16_t)(d + (uint16_t)dst[i - stride]);
    }
}

// Enchaîne les pré-traitements demandés : au plus deux étapes, (delta et/ou zigzag) puis (bit)shuffle.
// Les étapes alternent entre deux tampons, la dernière écrit dans dst.
const void *filters_forward(int flags, const int16_t *src, void *tmp, void *dst, size_t n, size_t stride) {
    int first = flags & (F_DELTA | F_ZIGZAG), second = flags & (F_SHUFFLE | F_BITSHUFFLE);
    if (!first && !second) return src;
    if (first == (F_DELTA | F_ZIGZAG) && second == F_BITSHUFFLE && !(flags & F_SCALAR)) {
        delta_zigzag_bitshuffle16(src, dst, tmp, n, stride);  // version fusionnée
        return dst;
    }
    void *mid = second ? tmp : dst;
    const void *cur = src;

    if (first == (F_DELTA | F_ZIGZAG)) delta_zigzag_forward(src, mid, n, stride), cur = mid;
    else if (first == F_DELTA) delta_forward(src, mid, n, stride), cur = mid;
    else if (first == F_ZIGZAG) zigzag_forward(src, mid, n), cur = mid;

    if (second == F_SHUFFLE) shuffle2(cur, dst, n), cur = dst;
    else if (second == F_BITSHUFFLE)
        (flags & F_SCALAR ? bitshuffle16_scalar : bitshuffle16)(cur, dst, n), cur = dst;
    return cur;
}

void filters_inverse(int flags, const void *src, void *tmp, int16_t *out, size_t n, size_t stride) {
    int first = flags & (F_DELTA | F_ZIGZAG), second = flags & (F_SHUFFLE | F_BITSHUFFLE);
    if (!first && !second) {
        memcpy(out, src, n * sizeof(int16_t));
        return;
    }
    const void *cur = src;
    void *mid = first ? tmp : out;  // sans delta ni zigzag, le (bit)unshuffle écrit directement dans out
    if (second == F_SHUFFLE) unshuffle2(cur, mid, n), cur = mid;
    else if (second == F_BITSHUFFLE)
        (flags & F_SCALAR ? bitunshuffle16_scalar : bitunshuffle16)(cur, mid, n), cur = mid;

    if (first == (F_DELTA | F_ZIGZAG)) delta_zigzag_inverse(cur, out, n, stride);
    else if (first == F_DELTA) delta_inverse(cur, out, n, stride);
    else if (first == F_ZIGZAG) zigzag_inverse(cur, out, n);
}
