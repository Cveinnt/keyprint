/* Development-only batched fixed-key HMAC-SHA256 experiment.
 * No SDK integration. EVP primitives implement SHA; no custom hash rounds.
 * Address bytes and output layout are defined by the Python caller.
 */
#include <stddef.h>
#include <stdint.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>

const char *keyprint_prf_openssl_version(void) {
    return OpenSSL_version(OPENSSL_VERSION);
}

int keyprint_prf_batch(const unsigned char *key, size_t key_len,
                      const unsigned char *prefix, size_t prefix_len,
                      const unsigned char *suffixes, size_t suffixes_len,
                      const size_t *offsets, size_t offset_count,
                      size_t layers, unsigned char *output, size_t output_len) {
    /* Validate every bound before accessing message data or writing output. */
    if (!key || key_len != 32 || !prefix || prefix_len > 65536 ||
        !suffixes || suffixes_len > 1048576 || !offsets ||
        offset_count < 2 || offset_count > 1001 ||
        layers < 1 || layers > 64 || !output) return 0;
    size_t count = offset_count - 1;
    size_t required = count * layers * 32;
    if (output_len != required || offsets[0] != 0 ||
        offsets[count] != suffixes_len) return 0;
    for (size_t i = 0; i < count; i++) {
        if (offsets[i] > offsets[i+1] || offsets[i+1] > suffixes_len ||
            offsets[i+1] - offsets[i] > 4096) return 0;
    }
    int ok = 0;
    unsigned char ipad[64] = {0}, opad[64] = {0}, inner_hash[32] = {0};
    unsigned char layer_bytes[4];
    unsigned int size = 0;
    EVP_MD *algorithm = EVP_MD_fetch(NULL, "SHA256", NULL);
    EVP_MD_CTX *base = EVP_MD_CTX_new(), *outer = EVP_MD_CTX_new();
    EVP_MD_CTX *layer_template = EVP_MD_CTX_new();
    EVP_MD_CTX *inner_work = EVP_MD_CTX_new(), *outer_work = EVP_MD_CTX_new();
    if (!algorithm || !base || !outer || !layer_template || !inner_work || !outer_work) goto done;
    for (size_t i = 0; i < 64; i++) {
        unsigned char b = i < 32 ? key[i] : 0;
        ipad[i] = b ^ 0x36;
        opad[i] = b ^ 0x5c;
    }
    if (EVP_DigestInit_ex2(base, algorithm, NULL) != 1 ||
        EVP_DigestUpdate(base, ipad, sizeof(ipad)) != 1 ||
        EVP_DigestUpdate(base, prefix, prefix_len) != 1 ||
        EVP_DigestInit_ex2(outer, algorithm, NULL) != 1 ||
        EVP_DigestUpdate(outer, opad, sizeof(opad)) != 1) goto done;
    for (size_t layer = 0; layer < layers; layer++) {
        layer_bytes[0] = (unsigned char)(layer >> 24);
        layer_bytes[1] = (unsigned char)(layer >> 16);
        layer_bytes[2] = (unsigned char)(layer >> 8);
        layer_bytes[3] = (unsigned char)layer;
        if (EVP_MD_CTX_copy_ex(layer_template, base) != 1 ||
            EVP_DigestUpdate(layer_template, layer_bytes, sizeof(layer_bytes)) != 1) goto done;
        for (size_t label = 0; label < count; label++) {
            size_t len = offsets[label+1] - offsets[label];
            if (EVP_MD_CTX_copy_ex(inner_work, layer_template) != 1 ||
                EVP_DigestUpdate(inner_work, suffixes + offsets[label], len) != 1 ||
                EVP_DigestFinal_ex(inner_work, inner_hash, &size) != 1 || size != 32 ||
                EVP_MD_CTX_copy_ex(outer_work, outer) != 1 ||
                EVP_DigestUpdate(outer_work, inner_hash, sizeof(inner_hash)) != 1 ||
                EVP_DigestFinal_ex(outer_work, output + (label * layers + layer) * 32, &size) != 1 ||
                size != 32) goto done;
        }
    }
    ok = 1;
done:
    if (!ok) OPENSSL_cleanse(output, required);
    OPENSSL_cleanse(ipad, sizeof(ipad));
    OPENSSL_cleanse(opad, sizeof(opad));
    OPENSSL_cleanse(inner_hash, sizeof(inner_hash));
    EVP_MD_CTX_free(base);
    EVP_MD_CTX_free(outer);
    EVP_MD_CTX_free(layer_template);
    EVP_MD_CTX_free(inner_work);
    EVP_MD_CTX_free(outer_work);
    EVP_MD_free(algorithm);
    return ok;
}
