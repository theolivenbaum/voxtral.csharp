using System;
using System.Collections.Generic;
using System.Numerics.Tensors;

namespace Voxtral
{
    public class Decoder
    {
        private const int LAYERS = 26;
        private const int DIM = 3072;
        private const int HEADS = 32;
        private const int HEAD_DIM = 128;
        private const int KV_HEADS = 8;
        private const int HIDDEN = 9216;
        private const int WINDOW = 8192;
        private const float NORM_EPS = 1e-5f;
        private const float ROPE_THETA = 1000000.0f;
        private const int VOCAB_SIZE = 131072;

        private readonly float[] _tokEmbeddings; // [VOCAB, DIM]
        private readonly float[] _finalNorm;
        private readonly DecoderLayer[] _layers;

        public Decoder(SafetensorsReader reader)
        {
            _tokEmbeddings = reader.LoadTensor("mm_streams_embeddings.embedding_module.tok_embeddings.weight").ToArray();
            _finalNorm = reader.LoadTensor("norm.weight").ToArray();

            _layers = new DecoderLayer[LAYERS];
            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i] = new DecoderLayer(reader, i);
            }
        }

        public float[] EmbedToken(int tokenId)
        {
            float[] embed = new float[DIM];
            Array.Copy(_tokEmbeddings, tokenId * DIM, embed, 0, DIM);
            return embed;
        }

        public void Prefill(float[] inputEmbeds, float[] tCond)
        {
            // inputEmbeds: [seq * DIM]
            int seqLen = inputEmbeds.Length / DIM;

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(inputEmbeds.AsSpan(), 0, seqLen, tCond);
            }
        }

        public float[] ForwardOne(float[] embed, int pos, float[] tCond)
        {
            // embed: [DIM]
            float[] h = new float[DIM];
            Array.Copy(embed, h, DIM);

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(h, pos, 1, tCond);
            }

            // Final Norm
            TensorOperations.RMSNorm(h, _finalNorm, h, NORM_EPS);

            // Logits: h @ tok_embeddings.T
            float[] logits = new float[VOCAB_SIZE];
            TensorOperations.Linear(h.AsSpan(), _tokEmbeddings, ReadOnlySpan<float>.Empty, logits, 1, VOCAB_SIZE, DIM);

            return logits;
        }

        public static float[] ComputeTimeEmbedding(float t, int dim)
        {
            float[] emb = new float[dim];
            int halfDim = dim / 2;
            float theta = 10000.0f;

            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)i / halfDim);
                float val = t * freq;
                emb[i] = MathF.Cos(val);
                emb[halfDim + i] = MathF.Sin(val);
            }
            return emb;
        }
    }

    class DecoderLayer
    {
        private const int DIM = 3072;
        private const int HEADS = 32;
        private const int HEAD_DIM = 128;
        private const int KV_HEADS = 8;
        private const int HIDDEN = 9216;
        private const int WINDOW = 8192;
        private const float NORM_EPS = 1e-5f;
        private const float ROPE_THETA = 1000000.0f;

        private readonly float[] _ropeFreqs;
        private readonly float[] _attnNorm, _ffnNorm;
        private readonly float[] _wq, _wk, _wv, _wo;
        private readonly float[] _w1, _w2, _w3;
        private readonly float[] _adaDown, _adaUp;

        // KV Cache
        private readonly float[] _kCache; // [WINDOW * KV_HEADS * HEAD_DIM]
        private readonly float[] _vCache; // [WINDOW * KV_HEADS * HEAD_DIM]

        public DecoderLayer(SafetensorsReader reader, int layerIdx)
        {
            // Precompute RoPE frequencies
            int halfDim = HEAD_DIM / 2;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                _ropeFreqs[i] = 1.0f / MathF.Pow(ROPE_THETA, 2.0f * i / HEAD_DIM);
            }

            string p = $"layers.{layerIdx}";

            _attnNorm = reader.LoadTensor($"{p}.attention_norm.weight").ToArray();
            _ffnNorm = reader.LoadTensor($"{p}.ffn_norm.weight").ToArray();

            _wq = reader.LoadTensor($"{p}.attention.wq.weight").ToArray();
            _wk = reader.LoadTensor($"{p}.attention.wk.weight").ToArray();
            _wv = reader.LoadTensor($"{p}.attention.wv.weight").ToArray();
            _wo = reader.LoadTensor($"{p}.attention.wo.weight").ToArray();

            _w1 = reader.LoadTensor($"{p}.feed_forward.w1.weight").ToArray();
            _w2 = reader.LoadTensor($"{p}.feed_forward.w2.weight").ToArray();
            _w3 = reader.LoadTensor($"{p}.feed_forward.w3.weight").ToArray();

            _adaDown = reader.LoadTensor($"{p}.ada_rms_norm_t_cond.0.weight").ToArray();
            _adaUp = reader.LoadTensor($"{p}.ada_rms_norm_t_cond.2.weight").ToArray();

            _kCache = new float[WINDOW * KV_HEADS * HEAD_DIM];
            _vCache = new float[WINDOW * KV_HEADS * HEAD_DIM];
        }

        public void Forward(Span<float> h, int pos, int seqLen, float[] tCond)
        {
            float[] xNorm = new float[seqLen * DIM];

            // 1. RMSNorm
            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(h.Slice(s*DIM, DIM), _attnNorm, xNorm.AsSpan(s*DIM, DIM), NORM_EPS);
            }

            // 2. Attention
            int qDim = HEADS * HEAD_DIM; // 4096
            int kvDim = KV_HEADS * HEAD_DIM; // 1024

            float[] q = new float[seqLen * qDim];
            float[] k = new float[seqLen * kvDim];
            float[] v = new float[seqLen * kvDim];

            TensorOperations.Linear(xNorm, _wq, ReadOnlySpan<float>.Empty, q, seqLen, qDim, DIM);
            TensorOperations.Linear(xNorm, _wk, ReadOnlySpan<float>.Empty, k, seqLen, kvDim, DIM);
            TensorOperations.Linear(xNorm, _wv, ReadOnlySpan<float>.Empty, v, seqLen, kvDim, DIM);

            // RoPE
            int halfDim = HEAD_DIM / 2;
            int ropeSize = seqLen * halfDim;
            // Use stackalloc for small sequences to avoid allocations
            Span<float> cos = ropeSize <= 4096 ? stackalloc float[ropeSize] : new float[ropeSize];
            Span<float> sin = ropeSize <= 4096 ? stackalloc float[ropeSize] : new float[ropeSize];

            ComputeRope(pos, seqLen, cos, sin);

            TensorOperations.ApplyRoPE(q, cos, sin, seqLen, HEADS, HEAD_DIM);
            TensorOperations.ApplyRoPE(k, cos, sin, seqLen, KV_HEADS, HEAD_DIM);

            // Update KV Cache
            UpdateKVCache(k, v, pos, seqLen);

            // Attention
            float[] attnOut = new float[seqLen * qDim]; // 4096
            PerformAttention(q, attnOut, pos, seqLen);

            // Output Projection
            float[] projOut = new float[seqLen * DIM];
            TensorOperations.Linear(attnOut, _wo, ReadOnlySpan<float>.Empty, projOut, seqLen, DIM, qDim);

            TensorPrimitives.Add(h, projOut, h);

            // 3. FFN
            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(h.Slice(s*DIM, DIM), _ffnNorm, xNorm.AsSpan(s*DIM, DIM), NORM_EPS);
            }

            // Ada Norm
            if (tCond != null)
            {
                float[] adaHidden = new float[32];
                TensorOperations.Linear(tCond, _adaDown, ReadOnlySpan<float>.Empty, adaHidden, 1, 32, DIM);
                TensorOperations.Gelu(adaHidden, adaHidden);

                float[] adaScale = new float[DIM];
                TensorOperations.Linear(adaHidden, _adaUp, ReadOnlySpan<float>.Empty, adaScale, 1, DIM, 32);

                for (int s = 0; s < seqLen; s++)
                {
                    var xn = xNorm.AsSpan(s * DIM, DIM);
                    for (int d = 0; d < DIM; d++)
                    {
                        xn[d] *= (1.0f + adaScale[d]);
                    }
                }
            }

            // SwiGLU
            float[] gate = new float[seqLen * HIDDEN];
            float[] up = new float[seqLen * HIDDEN];

            TensorOperations.Linear(xNorm, _w1, ReadOnlySpan<float>.Empty, gate, seqLen, HIDDEN, DIM);
            TensorOperations.Linear(xNorm, _w3, ReadOnlySpan<float>.Empty, up, seqLen, HIDDEN, DIM);

            TensorOperations.SiLU(gate, gate);
            TensorPrimitives.Multiply(gate, up, gate);

            TensorOperations.Linear(gate, _w2, ReadOnlySpan<float>.Empty, projOut, seqLen, DIM, HIDDEN);

            TensorPrimitives.Add(h, projOut, h);
        }

        private void ComputeRope(int pos, int seqLen, Span<float> cos, Span<float> sin)
        {
            int halfDim = HEAD_DIM / 2;

            for (int s = 0; s < seqLen; s++)
            {
                int absPos = pos + s;
                for (int i = 0; i < halfDim; i++)
                {
                    float angle = absPos * _ropeFreqs[i];
                    cos[s * halfDim + i] = MathF.Cos(angle);
                    sin[s * halfDim + i] = MathF.Sin(angle);
                }
            }
        }

        private void UpdateKVCache(float[] k, float[] v, int pos, int seqLen)
        {
            int dim = KV_HEADS * HEAD_DIM;

            for (int s = 0; s < seqLen; s++)
            {
                int absPos = pos + s;
                int cacheIdx = absPos % WINDOW;

                Array.Copy(k, s * dim, _kCache, cacheIdx * dim, dim);
                Array.Copy(v, s * dim, _vCache, cacheIdx * dim, dim);
            }
        }

        private void PerformAttention(float[] q, float[] output, int pos, int seqLen)
        {
            int qHeads = HEADS;
            int kvHeads = KV_HEADS;
            int gqaRatio = qHeads / kvHeads;
            int dim = HEAD_DIM;
            float scale = 1.0f / MathF.Sqrt(dim);

            Parallel.For(0, qHeads, h =>
            {
                int kvH = h / gqaRatio;

                for (int s = 0; s < seqLen; s++)
                {
                    int absPos = pos + s;
                    // Note: accessing q and output slices is thread-safe as they are distinct for each h
                    // Need to be careful with Span in lambda?
                    // Span cannot be captured in lambda if lambda is closure over local span.
                    // But q and output are arrays. We slice them inside the lambda. This is safe.

                    var qi = q.AsSpan((s * qHeads + h) * dim, dim);

                    int startP = Math.Max(0, absPos - WINDOW + 1);
                    int endP = absPos;
                    int len = endP - startP + 1;

                    // Allocate scores on stack if small?
                    // len <= WINDOW (8192). 32KB.
                    // Stackalloc in loop inside Parallel.For is dangerous if recursion or many iterations?
                    // But here seqLen is small.
                    // Use array for safety.
                    float[] scores = new float[len];

                    for (int j = 0; j < len; j++)
                    {
                        int pIdx = startP + j;
                        int cacheIdx = pIdx % WINDOW;
                        var kj = _kCache.AsSpan((cacheIdx * kvHeads + kvH) * dim, dim);

                        scores[j] = TensorPrimitives.Dot(qi, kj) * scale;
                    }

                    TensorOperations.Softmax(scores);

                    var outH = output.AsSpan((s * qHeads + h) * dim, dim);
                    outH.Clear();

                    for (int j = 0; j < len; j++)
                    {
                        int pIdx = startP + j;
                        int cacheIdx = pIdx % WINDOW;
                        var vj = _vCache.AsSpan((cacheIdx * kvHeads + kvH) * dim, dim);
                        float score = scores[j];

                        for (int d = 0; d < dim; d++)
                        {
                            outH[d] += score * vj[d];
                        }
                    }
                }
            });
        }
    }
}
