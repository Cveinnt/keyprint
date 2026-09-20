/* MIT. Bounded binary32 top-k selection. No global or keyed state. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t rank; uint32_t index; } Entry;

/* The root is the worst retained entry: low score, then high original index. */
static int worse(Entry a, Entry b) {
    return a.rank < b.rank || (a.rank == b.rank && a.index > b.index);
}

static int best_first(const void *left, const void *right) {
    Entry a = *(const Entry *)left, b = *(const Entry *)right;
    if (worse(a, b)) return 1;
    if (worse(b, a)) return -1;
    return 0;
}

int keyprint_select_f32(const unsigned char *scores, size_t scores_len,
                        size_t top_k, uint32_t *output, size_t output_len) {
    if (!scores || !output || scores_len % 4 || scores_len < 4 ||
        scores_len > 151669 * 4 || top_k < 1 || top_k > 512 ||
        top_k > scores_len / 4 || output_len != top_k * sizeof(uint32_t)) return 0;
    Entry heap[512];
    size_t count = scores_len / 4, used = 0;
    for (size_t index = 0; index < count; index++) {
        const unsigned char *p = scores + index * 4;
        uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        if ((bits & 0x7f800000u) == 0x7f800000u) return 0;
        /* Integer ranking preserves subnormals even under host flush-to-zero
         * modes. Both signed zeros compare equal; indices break their ties. */
        if (!(bits & 0x7fffffffu)) bits = 0;
        uint32_t rank = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
        Entry entry = {rank, (uint32_t)index};
        if (used < top_k) {
            size_t child = used++;
            heap[child] = entry;
            while (child) {
                size_t parent = (child - 1) / 2;
                if (!worse(heap[child], heap[parent])) break;
                Entry temp = heap[parent]; heap[parent] = heap[child]; heap[child] = temp;
                child = parent;
            }
        } else if (worse(heap[0], entry)) {
            heap[0] = entry;
            size_t parent = 0;
            while (parent * 2 + 1 < used) {
                size_t child = parent * 2 + 1;
                if (child + 1 < used && worse(heap[child + 1], heap[child])) child++;
                if (!worse(heap[child], heap[parent])) break;
                Entry temp = heap[parent]; heap[parent] = heap[child]; heap[child] = temp;
                parent = child;
            }
        }
    }
    qsort(heap, used, sizeof(Entry), best_first);
    /* No output writes until every input has passed validation. */
    for (size_t i = 0; i < used; i++) output[i] = heap[i].index;
    return 1;
}
