# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Current Baseline (2026-02-06)
- Decoder: 43.2 ms/token
- Encoder: ~2.4s (test_speech.wav, 3.6s audio)
- Theoretical floor: ~23 ms/token (300 GB/s bandwidth, 6.9 GB weights)
- Remaining overhead: ~79 command buffer round-trips per token

## Already Optimized
- Fused QKV: 3 matmuls → 1 command buffer (saves 52 round-trips/token)
- Fused FFN: w1+w3+silu+mul+w2 → 1 command buffer (saves 52 round-trips/token)
- Persistent decoder buffers: eliminates 11 malloc/free per token
- Pre-warm weight cache: all bf16→f16 conversion at load time (8.4 GB GPU)
- GPU batched encoder attention: causal_softmax shader + strided MPS views
- Fused encoder QKV + FFN (reusing decoder fused functions)

## Optimization Attempts

### Attempt 1: Fused wo+FFN + Fused norm+QKV + Fused logits (SUCCESS)
- **Fused wo+FFN**: wo projection + residual + rms_norm + ada_scale + SwiGLU FFN + residual in 1 cmd buf
  - Eliminates separate wo cmd buf per layer
  - Result: 43 → ~40 ms/step
- **Fused norm+QKV**: rms_norm + 3 QKV matmuls in 1 cmd buf (replaces separate norm + fused QKV)
  - Marginal additional improvement (CPU rms_norm was fast)
- **Fused logits**: final rms_norm + logits matmul + argmax_f32 shader in 1 cmd buf
  - Avoids 512KB logits download when only argmax needed
  - Result: ~40 → ~38 ms/step
- **Prefill timing split**: separate prefill_ms from per-step timing for accurate measurement
- **Combined: 43.2 → 38 ms/step (12% improvement)**
- Command buffers per token: 79 → 53 (26×2 per layer + 1 logits)
- New shaders: ada_scale_mul, argmax_f32

### Attempt 2: Cross-layer fusion — persistent GPU x (SUCCESS)
- Keep x on GPU buffer across all 26 layers (no CPU round-trip between layers)
- Fuse wo_ffn[i] + norm_qkv[i+1] into a single command buffer
- Helper functions: encode_wo_ffn_steps(), encode_norm_qkv_steps()
- New API: decoder_start/end, decoder_norm_qkv, decoder_wo_ffn_next_qkv, decoder_wo_ffn_logits
- Command buffers per token: 53 → 27 (1 + 25 fused + 1 final)
- **Result: 38 → 32.5 ms/step (test_speech), 39 → 35.7 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 32.5 ms/step (25% faster)**

### Next targets
- Theoretical floor: ~23 ms (300 GB/s, 6.9 GB weights)
- Remaining: 27 cmd bufs × ~0.3ms = ~8ms overhead
- Gap: 32.5 - 23 = 9.5ms (some is cmd buf overhead, some is GPU compute overhead)
- Ideas: GPU RoPE + GPU attention → fuse entire layer into 1 cmd buf (27 → 1)
