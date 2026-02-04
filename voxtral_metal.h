/*
 * voxtral_metal.h - Metal GPU acceleration for Voxtral inference
 *
 * Provides MPS-accelerated matrix multiplication with bf16->f16 weight caching,
 * plus GPU compute shaders for element-wise operations.
 * Ported from flux-2-4b (same author, same license).
 */

#ifndef VOXTRAL_METAL_H
#define VOXTRAL_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Metal acceleration. Returns 1 on success, 0 if unavailable. */
int vox_metal_init(void);

/* Check if Metal is initialized and available. */
int vox_metal_available(void);

/* Cleanup all Metal resources. */
void vox_metal_shutdown(void);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = alpha * A[M,K] @ B^T[N,K] + beta * C[M,N]
 *
 * A is f32 (activations), B is bf16 (weights, converted to f16 for MPS),
 * C is f32 (output). B is always transposed (row-major weight layout).
 *
 * Weight buffers are cached on GPU after first use (bf16->f16 conversion
 * happens once per unique weight pointer).
 */
void vox_metal_sgemm_bf16(int M, int N, int K,
                           const float *A,
                           const uint16_t *B_bf16,
                           float *C);

/*
 * GPU-accelerated f32 matrix multiplication.
 * C[M,N] = A[M,K] @ B^T[N,K]
 */
void vox_metal_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C);

/*
 * Fused QKV: three matmuls in one command buffer with shared input.
 * q[M,Nq] = input[M,K] @ wq[Nq,K]^T
 * k[M,Nk] = input[M,K] @ wk[Nk,K]^T
 * v[M,Nv] = input[M,K] @ wv[Nv,K]^T
 * Saves 2 command buffer round-trips vs 3 separate calls.
 */
void vox_metal_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q, float *k, float *v);

/*
 * Fused SwiGLU FFN: w1+w3+silu+mul+w2 in one command buffer.
 * gate = silu(input @ w1^T)
 * up = input @ w3^T
 * output = (gate * up) @ w2^T
 * All intermediate data stays on GPU. Saves 2 round-trips + eliminates
 * intermediate CPU memcpy for silu/mul.
 */
void vox_metal_fused_ffn_bf16(int M, int dim, int hidden,
                               const float *input,
                               const uint16_t *w1_bf16,
                               const uint16_t *w3_bf16,
                               const uint16_t *w2_bf16,
                               float *output);

/*
 * GPU batched attention (all heads in one command buffer).
 * Performs QK^T matmul, causal+window masked softmax, scores*V matmul
 * entirely on GPU. Uses strided MPS matrix views (no per-head copies).
 *
 * Q:   [seq_q, n_heads * head_dim]   f32
 * K:   [seq_k, n_kv_heads * head_dim] f32
 * V:   [seq_k, n_kv_heads * head_dim] f32
 * out: [seq_q, n_heads * head_dim]   f32
 */
void vox_metal_batched_attention(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, float scale,
                                  int window_size, int q_offset);

/*
 * Pre-warm the bf16->f16 cache for a weight tensor.
 * Call during model loading to avoid first-use latency.
 */
void vox_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/* GPU memory usage (for debugging). */
size_t vox_metal_memory_used(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_METAL_H */
