/*
 * voxtral_encoder.c - Audio encoder (causal transformer)
 *
 * Architecture:
 *   Conv stem: conv1d(128->1280, k=3, s=1, p=1) -> GELU
 *              conv1d(1280->1280, k=3, s=2, p=1) -> GELU
 *   32 transformer layers (causal, sliding window=750):
 *     - RMSNorm -> Attention (MHA, 32 heads, head_dim=64, with biases)
 *     - RMSNorm -> SwiGLU FFN (dim=1280, hidden=5120, w2 has bias)
 *   Final RMSNorm
 *   Downsample 4x: reshape [seq, 1280] -> [seq/4, 5120]
 *   Adapter: Linear(5120->3072) -> GELU -> Linear(3072->3072)
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_safetensors.h"
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

#define ENC_PREFIX "mm_streams_embeddings.embedding_module.whisper_encoder"

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

int vox_encoder_load(vox_encoder_t *enc, safetensors_file_t *sf) {
    char name[512];

    /* Conv stem (small, always f32) */
    snprintf(name, sizeof(name), "%s.conv_layers.0.conv.weight", ENC_PREFIX);
    enc->conv0_weight = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv_layers.0.conv.bias", ENC_PREFIX);
    enc->conv0_bias = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv_layers.1.conv.weight", ENC_PREFIX);
    enc->conv1_weight = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv_layers.1.conv.bias", ENC_PREFIX);
    enc->conv1_bias = load_f32(sf, name);

    if (!enc->conv0_weight || !enc->conv1_weight) return -1;

    /* Transformer layers */
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_t *l = &enc->layers[i];
        const char *lp = ENC_PREFIX ".transformer.layers";

        /* Large matmul weights: bf16 mmap direct */
        snprintf(name, sizeof(name), "%s.%d.attention.wq.weight", lp, i);
        l->wq_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention.wk.weight", lp, i);
        l->wk_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention.wv.weight", lp, i);
        l->wv_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention.wo.weight", lp, i);
        l->wo_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w1.weight", lp, i);
        l->w1_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w2.weight", lp, i);
        l->w2_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w3.weight", lp, i);
        l->w3_weight_bf16 = load_bf16_direct(sf, name);

        /* Small weights: biases and norms (always f32) */
        snprintf(name, sizeof(name), "%s.%d.attention.wq.bias", lp, i);
        l->wq_bias = load_f32(sf, name);
        /* wk has NO bias */
        snprintf(name, sizeof(name), "%s.%d.attention.wv.bias", lp, i);
        l->wv_bias = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention.wo.bias", lp, i);
        l->wo_bias = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention_norm.weight", lp, i);
        l->attention_norm = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w2.bias", lp, i);
        l->w2_bias = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.ffn_norm.weight", lp, i);
        l->ffn_norm = load_f32(sf, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16) {
            fprintf(stderr, "encoder: failed to load layer %d weights\n", i);
            return -1;
        }

        if (vox_verbose >= 2)
            fprintf(stderr, "  Encoder layer %d/%d loaded\n", i + 1, VOX_ENC_LAYERS);
    }

    /* Final norm */
    snprintf(name, sizeof(name), "%s.transformer.norm.weight", ENC_PREFIX);
    enc->norm = load_f32(sf, name);

    if (!enc->norm) return -1;
    return 0;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* GELU activation */
static void gelu_inplace(float *x, int n) {
    vox_gelu(x, n);
}

static int causal_conv1d_out_len(int length, int kernel_size, int stride) {
    int padding_total = kernel_size - stride;
    float n_frames = ((float)length - kernel_size + padding_total) / (float)stride + 1.0f;
    int out_len = (int)ceilf(n_frames);
    return out_len < 0 ? 0 : out_len;
}

float *vox_encoder_forward(vox_ctx_t *ctx, const float *mel,
                           int mel_frames, int *out_seq_len) {
    vox_encoder_t *enc = &ctx->encoder;
    int dim = VOX_ENC_DIM;        /* 1280 */
    int n_heads = VOX_ENC_HEADS;  /* 32 */
    int head_dim = VOX_ENC_HEAD_DIM; /* 64 */
    int hidden = VOX_ENC_HIDDEN;  /* 5120 */
    int qkv_dim = n_heads * head_dim; /* 2048 */

    if (vox_verbose >= 2)
        fprintf(stderr, "Encoder: %d mel frames\n", mel_frames);

    /* ---- Conv stem ---- */
    /* mel: [mel_frames, 128] -> transpose to [128, mel_frames] for conv1d */
    float *conv_in = (float *)malloc(VOX_MEL_BINS * mel_frames * sizeof(float));
    for (int f = 0; f < mel_frames; f++) {
        for (int m = 0; m < VOX_MEL_BINS; m++) {
            conv_in[m * mel_frames + f] = mel[f * VOX_MEL_BINS + m];
        }
    }

    /* Conv0: [128, mel_frames] -> [1280, mel_frames] (stride=1, causal) */
    int conv0_out_len = causal_conv1d_out_len(mel_frames, 3, 1);
    float *conv0_out = (float *)malloc(dim * conv0_out_len * sizeof(float));
    vox_causal_conv1d(conv0_out, conv_in, enc->conv0_weight, enc->conv0_bias,
                      VOX_MEL_BINS, dim, mel_frames, 3, 1);
    gelu_inplace(conv0_out, dim * conv0_out_len);
    free(conv_in);

    /* Conv1: [1280, mel_frames] -> [1280, ceil(mel_frames/2)] (stride=2, causal) */
    int conv1_out_len = causal_conv1d_out_len(conv0_out_len, 3, 2);
    float *conv1_out = (float *)malloc(dim * conv1_out_len * sizeof(float));
    vox_causal_conv1d(conv1_out, conv0_out, enc->conv1_weight, enc->conv1_bias,
                      dim, dim, conv0_out_len, 3, 2);
    gelu_inplace(conv1_out, dim * conv1_out_len);
    free(conv0_out);

    int seq_len = conv1_out_len;

    /* Transpose: [1280, seq_len] -> [seq_len, 1280] */
    float *x = (float *)malloc(seq_len * dim * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            x[s * dim + d] = conv1_out[d * seq_len + s];
        }
    }
    free(conv1_out);

    if (vox_verbose >= 2)
        fprintf(stderr, "  Conv stem: %d frames -> %d\n", mel_frames, seq_len);

    /* ---- Transformer layers ---- */
    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    float *q = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *k = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *v = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *attn_out = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    float *gate = (float *)malloc(seq_len * hidden * sizeof(float));
    float *up = (float *)malloc(seq_len * hidden * sizeof(float));
    float *ffn_out = (float *)malloc(seq_len * dim * sizeof(float));

    /* RoPE frequencies */
    int *positions = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) positions[i] = i;
    float *rope_freqs = (float *)malloc(seq_len * (head_dim / 2) * 2 * sizeof(float));
    vox_compute_rope_freqs(rope_freqs, positions, seq_len, head_dim, VOX_ROPE_THETA);

    for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
        vox_enc_layer_t *l = &enc->layers[layer];

        /* ---- Self-attention ---- */
        vox_rms_norm(x_norm, x, l->attention_norm, seq_len, dim, VOX_ENC_NORM_EPS);

        /* Q, K, V projections (bf16 weights, f32 biases) */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_qkv_bf16(seq_len, dim, x_norm,
                                      l->wq_weight_bf16, qkv_dim,
                                      l->wk_weight_bf16, qkv_dim,
                                      l->wv_weight_bf16, qkv_dim,
                                      q, k, v);
            /* Add biases on CPU (wq has bias, wk has NO bias, wv has bias) */
            for (int s = 0; s < seq_len; s++) {
                for (int j = 0; j < qkv_dim; j++) {
                    q[s * qkv_dim + j] += l->wq_bias[j];
                    v[s * qkv_dim + j] += l->wv_bias[j];
                }
            }
        } else {
#endif
            vox_linear_bf16(q, x_norm, l->wq_weight_bf16, l->wq_bias, seq_len, dim, qkv_dim);
            vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, qkv_dim);
            vox_linear_bf16(v, x_norm, l->wv_weight_bf16, l->wv_bias, seq_len, dim, qkv_dim);
#ifdef USE_METAL
        }
#endif

        /* Apply RoPE to Q and K */
        vox_apply_rope(q, rope_freqs, seq_len, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, seq_len, n_heads, head_dim);

        /* Causal attention with sliding window */
        float scale = 1.0f / sqrtf((float)head_dim);
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_batched_attention(attn_out, q, k, v,
                                         seq_len, seq_len, n_heads, VOX_ENC_KV_HEADS,
                                         head_dim, scale, VOX_ENC_WINDOW, 0);
        } else {
#endif
            vox_causal_attention(attn_out, q, k, v,
                                 seq_len, seq_len, n_heads, VOX_ENC_KV_HEADS,
                                 head_dim, scale, VOX_ENC_WINDOW, 0);
#ifdef USE_METAL
        }
#endif

        /* Output projection + residual */
        vox_linear_bf16(proj_out, attn_out, l->wo_weight_bf16, l->wo_bias, seq_len, qkv_dim, dim);
        vox_add_inplace(x, proj_out, seq_len * dim);

        /* ---- FFN ---- */
        vox_rms_norm(x_norm, x, l->ffn_norm, seq_len, dim, VOX_ENC_NORM_EPS);

        /* SwiGLU: gate = silu(w1(x)), up = w3(x), ffn = w2(gate * up) + bias */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_ffn_bf16(seq_len, dim, hidden, x_norm,
                                      l->w1_weight_bf16, l->w3_weight_bf16,
                                      l->w2_weight_bf16, ffn_out);
            /* Add w2 bias on CPU */
            for (int s = 0; s < seq_len; s++)
                for (int j = 0; j < dim; j++)
                    ffn_out[s * dim + j] += l->w2_bias[j];
        } else {
#endif
            vox_linear_nobias_bf16(gate, x_norm, l->w1_weight_bf16, seq_len, dim, hidden);
            vox_silu(gate, seq_len * hidden);
            vox_linear_nobias_bf16(up, x_norm, l->w3_weight_bf16, seq_len, dim, hidden);
            vox_mul_inplace(gate, up, seq_len * hidden);
            vox_linear_bf16(ffn_out, gate, l->w2_weight_bf16, l->w2_bias, seq_len, hidden, dim);
#ifdef USE_METAL
        }
#endif

        /* Residual */
        vox_add_inplace(x, ffn_out, seq_len * dim);

        if (vox_verbose >= 2 && ((layer + 1) % 8 == 0 || layer == VOX_ENC_LAYERS - 1))
            fprintf(stderr, "  Encoder layer %d/%d\n", layer + 1, VOX_ENC_LAYERS);
    }

    /* Final norm */
    vox_rms_norm(x, x, enc->norm, seq_len, dim, VOX_ENC_NORM_EPS);

    /* Clean up working buffers */
    free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out);
    free(gate); free(up); free(ffn_out);
    free(positions); free(rope_freqs);

    *out_seq_len = seq_len;
    return x;
}

/* ========================================================================
 * Adapter Forward Pass
 * ======================================================================== */

float *vox_adapter_forward(vox_ctx_t *ctx, const float *enc_out,
                           int enc_seq_len, int *out_seq_len) {
    /* Downsample 4x: [enc_seq_len, 1280] -> [enc_seq_len/4, 5120] */
    int ds_len = enc_seq_len / VOX_DOWNSAMPLE;
    int ds_dim = VOX_ENC_DIM * VOX_DOWNSAMPLE; /* 5120 */

    float *ds = (float *)malloc(ds_len * ds_dim * sizeof(float));
    for (int i = 0; i < ds_len; i++) {
        for (int j = 0; j < VOX_DOWNSAMPLE; j++) {
            memcpy(ds + i * ds_dim + j * VOX_ENC_DIM,
                   enc_out + (i * VOX_DOWNSAMPLE + j) * VOX_ENC_DIM,
                   VOX_ENC_DIM * sizeof(float));
        }
    }

    if (vox_verbose >= 2)
        fprintf(stderr, "  Adapter: %d -> %d (downsample %dx)\n",
                enc_seq_len, ds_len, VOX_DOWNSAMPLE);

    /* Linear(5120 -> 3072) -> GELU -> Linear(3072 -> 3072) */
    float *mid = (float *)malloc(ds_len * VOX_DEC_DIM * sizeof(float));
    vox_linear_nobias_bf16(mid, ds, ctx->adapter.linear0_weight_bf16, ds_len, ds_dim, VOX_DEC_DIM);
    vox_gelu(mid, ds_len * VOX_DEC_DIM);

    float *out = (float *)malloc(ds_len * VOX_DEC_DIM * sizeof(float));
    vox_linear_nobias_bf16(out, mid, ctx->adapter.linear1_weight_bf16, ds_len, VOX_DEC_DIM, VOX_DEC_DIM);

    free(ds);
    free(mid);

    *out_seq_len = ds_len;
    return out;
}
