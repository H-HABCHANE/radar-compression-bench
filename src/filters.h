// filters.h — pré-traitements réversibles appliqués avant la compression, sur des int16.
// Aucun n'apporte de gain seul : ils réorganisent les données pour que le compresseur
// trouve plus facilement les redondances.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Pré-traitements combinables, appliqués dans cet ordre (et défaits dans l'ordre inverse)
enum {
    F_DELTA = 1,      // x[i] - x[i - stride] : retire ce qui se répète d'une impulsion à l'autre
    F_ZIGZAG = 2,     // entier signé -> non signé (0, -1, 1, -2, ...) : petits nombres = bits forts à 0
    F_SHUFFLE = 4,    // regroupe les octets de même rang (tous les octets faibles, puis tous les forts)
    F_BITSHUFFLE = 8, // regroupe les bits de même rang : 16 plans de bits
    F_SCALAR = 16     // force le bitshuffle scalaire (pour mesurer le gain de la version AVX2)
};

// Applique les pré-traitements `flags` à n valeurs de src. tmp et dst ont la taille de src.
// stride : distance (en valeurs) entre deux échantillons à soustraire pour F_DELTA.
// Renvoie le tampon qui contient le résultat (dst ou tmp).
const void *filters_forward(int flags, const int16_t *src, void *tmp, void *dst, size_t n, size_t stride);

// Défait les pré-traitements : src contient le résultat de filters_forward, out reçoit les n valeurs.
void filters_inverse(int flags, const void *src, void *tmp, int16_t *out, size_t n, size_t stride);

// Fonctions élémentaires, exposées pour les tests
void delta_forward(const int16_t *src, int16_t *dst, size_t n, size_t stride);
void delta_inverse(const int16_t *src, int16_t *dst, size_t n, size_t stride);
void zigzag_forward(const int16_t *src, uint16_t *dst, size_t n);
void zigzag_inverse(const uint16_t *src, int16_t *dst, size_t n);
void shuffle2(const uint8_t *src, uint8_t *dst, size_t n);
void unshuffle2(const uint8_t *src, uint8_t *dst, size_t n);
void bitshuffle16(const uint16_t *src, uint8_t *dst, size_t n);        // AVX2 si disponible
void bitshuffle16_scalar(const uint16_t *src, uint8_t *dst, size_t n); // version portable
void bitunshuffle16(const uint8_t *src, uint16_t *dst, size_t n);        // AVX2 si disponible
void bitunshuffle16_scalar(const uint8_t *src, uint16_t *dst, size_t n); // version portable
void delta_zigzag_forward(const int16_t *src, uint16_t *dst, size_t n, size_t stride);
void delta_zigzag_inverse(const uint16_t *src, int16_t *dst, size_t n, size_t stride);
// Les trois en un seul passage sur la mémoire (tmp ne sert que si n ou stride n'est pas multiple de 32)
void delta_zigzag_bitshuffle16(const int16_t *src, uint8_t *dst, uint16_t *tmp, size_t n, size_t stride);
