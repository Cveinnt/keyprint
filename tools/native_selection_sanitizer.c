/* Memory/UB stress harness; independent numerical oracle lives in Python tests. */
#include <assert.h>
#include <stdio.h>
#include "../native/src/keyprint_native/_select.c"

int main(void) {
    unsigned char *allocation = malloc(151669 * 4 + 1);
    assert(allocation);
    unsigned char *scores = allocation + 1; /* deliberately unaligned */
    uint32_t seed = 20260920, output[514];
    for (size_t round = 0; round < 10000; round++) {
        size_t count = round % 100 == 0 ? 151669 : 1 + round % 2048;
        size_t k = 1 + round % 512;
        if (k > count) k = count;
        for (size_t i = 0; i < count; i++) {
            seed = seed * 1664525u + 1013904223u;
            uint32_t word = seed;
            if ((word & 0x7f800000u) == 0x7f800000u) word = 0;
            for (size_t b = 0; b < 4; b++) scores[i * 4 + b] = (unsigned char)(word >> (8 * b));
        }
        for (size_t i = 0; i < 514; i++) output[i] = 0xdeadbeefu;
        assert(keyprint_select_f32(scores, count * 4, k, output + 1, k * 4) == 1);
        assert(output[0] == 0xdeadbeefu);
        for (size_t i = k + 1; i < 514; i++) assert(output[i] == 0xdeadbeefu);
        for (size_t i = 1; i <= k; i++) {
            assert(output[i] < count);
            for (size_t j = 1; j < i; j++) assert(output[i] != output[j]);
        }
        /* Invalid final score must not expose any partially selected result. */
        scores[(count - 1) * 4 + 2] = 0x80;
        scores[(count - 1) * 4 + 3] = 0x7f;
        for (size_t i = 0; i < 514; i++) output[i] = 0xdeadbeefu;
        assert(keyprint_select_f32(scores, count * 4, k, output + 1, k * 4) == 0);
        for (size_t i = 0; i < 514; i++) assert(output[i] == 0xdeadbeefu);
    }
    free(allocation);
    puts("10000 valid and 10000 invalid selector calls passed memory guards");
    return 0;
}
