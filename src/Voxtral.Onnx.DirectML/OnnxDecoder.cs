using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx.DirectML
{
    public class OnnxDecoder
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

        private readonly DenseTensor<float> _tokEmbeddings; // [VOCAB, DIM]
        private readonly DenseTensor<float> _finalNorm;
        private readonly OnnxDecoderLayer[] _layers;

        public OnnxDecoder(OnnxSafetensorsReader reader)
        {
            _tokEmbeddings = reader.LoadTensor("mm_streams_embeddings.embedding_module.tok_embeddings.weight");
            _finalNorm = reader.LoadTensor("norm.weight");

            _layers = new OnnxDecoderLayer[LAYERS];
            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i] = new OnnxDecoderLayer(reader, i);
            }
        }

        public DenseTensor<float> EmbedToken(int tokenId)
        {
            float[] embedData = new float[DIM];
            DenseTensor<float> embed = new DenseTensor<float>(embedData, new int[] { DIM });

            var src = _tokEmbeddings.Buffer.Span.Slice(tokenId * DIM, DIM);
            src.CopyTo(embed.Buffer.Span);

            return embed;
        }

        public void Prefill(DenseTensor<float> inputEmbeds, DenseTensor<float> tCond)
        {
            int seqLen = (int)(inputEmbeds.Buffer.Length / DIM);

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(inputEmbeds, 0, seqLen, tCond);
            }
        }

        public DenseTensor<float> ForwardOne(DenseTensor<float> embed, int pos, DenseTensor<float> tCond)
        {
            float[] hData = new float[DIM];
            DenseTensor<float> h = new DenseTensor<float>(hData, new int[] { DIM });
            embed.Buffer.Span.CopyTo(h.Buffer.Span);

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(h, pos, 1, tCond);
            }

            OnnxTensorOperations.RMSNorm(h, _finalNorm, h, NORM_EPS);

            float[] logitsData = new float[VOCAB_SIZE];
            DenseTensor<float> logits = new DenseTensor<float>(logitsData, new int[] { VOCAB_SIZE });

            OnnxTensorOperations.Linear(h, _tokEmbeddings, null, logits);

            return logits;
        }

        public static DenseTensor<float> ComputeTimeEmbedding(float t, int dim)
        {
            float[] embData = new float[dim];
            int halfDim = dim / 2;
            float theta = 10000.0f;

            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)i / halfDim);
                float val = t * freq;
                embData[i] = MathF.Cos(val);
                embData[halfDim + i] = MathF.Sin(val);
            }
            return new DenseTensor<float>(embData, new int[] { dim });
        }
    }

    class OnnxDecoderLayer
    {
        private const int DIM = 3072;
        private const int HEADS = 32;
        private const int HEAD_DIM = 128;
        private const int KV_HEADS = 8;
        private const int HIDDEN = 9216;
        private const int WINDOW = 8192;
        private const float NORM_EPS = 1e-5f;
        private const float ROPE_THETA = 1000000.0f;

        private readonly DenseTensor<float> _ropeFreqs;
        private readonly DenseTensor<float> _attnNorm, _ffnNorm;
        private readonly DenseTensor<float> _wq, _wk, _wv, _wo;
        private readonly DenseTensor<float> _w1, _w2, _w3;
        private readonly DenseTensor<float> _adaDown, _adaUp;

        private readonly DenseTensor<float> _kCache; // [WINDOW, KV_HEADS, HEAD_DIM]
        private readonly DenseTensor<float> _vCache; // [WINDOW, KV_HEADS, HEAD_DIM]

        public OnnxDecoderLayer(OnnxSafetensorsReader reader, int layerIdx)
        {
            int halfDim = HEAD_DIM / 2;
            float[] ropeFreqsData = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                ropeFreqsData[i] = 1.0f / MathF.Pow(ROPE_THETA, 2.0f * i / HEAD_DIM);
            }
            _ropeFreqs = new DenseTensor<float>(ropeFreqsData, new int[] { halfDim });

            string p = $"layers.{layerIdx}";

            _attnNorm = reader.LoadTensor($"{p}.attention_norm.weight");
            _ffnNorm = reader.LoadTensor($"{p}.ffn_norm.weight");

            _wq = reader.LoadTensor($"{p}.attention.wq.weight");
            _wk = reader.LoadTensor($"{p}.attention.wk.weight");
            _wv = reader.LoadTensor($"{p}.attention.wv.weight");
            _wo = reader.LoadTensor($"{p}.attention.wo.weight");

            _w1 = reader.LoadTensor($"{p}.feed_forward.w1.weight");
            _w2 = reader.LoadTensor($"{p}.feed_forward.w2.weight");
            _w3 = reader.LoadTensor($"{p}.feed_forward.w3.weight");

            _adaDown = reader.LoadTensor($"{p}.ada_rms_norm_t_cond.0.weight");
            _adaUp = reader.LoadTensor($"{p}.ada_rms_norm_t_cond.2.weight");

            float[] kCacheData = new float[WINDOW * KV_HEADS * HEAD_DIM];
            float[] vCacheData = new float[WINDOW * KV_HEADS * HEAD_DIM];
            _kCache = new DenseTensor<float>(kCacheData, new int[] { WINDOW, KV_HEADS, HEAD_DIM });
            _vCache = new DenseTensor<float>(vCacheData, new int[] { WINDOW, KV_HEADS, HEAD_DIM });
        }

        public void Forward(DenseTensor<float> h, int pos, int seqLen, DenseTensor<float> tCond)
        {
            var hSpan = h.Buffer.Span;
            float[] xNormData = new float[seqLen * DIM];
            DenseTensor<float> xNorm = new DenseTensor<float>(xNormData, new int[] { seqLen, DIM });
            var xNormSpan = xNorm.Buffer.Span;
            var attnNormSpan = _attnNorm.Buffer.Span;

            for (int s = 0; s < seqLen; s++)
            {
                OnnxTensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), attnNormSpan, xNormSpan.Slice(s*DIM, DIM), NORM_EPS);
            }

            int qDim = HEADS * HEAD_DIM;
            int kvDim = KV_HEADS * HEAD_DIM;

            float[] qData = new float[seqLen * qDim];
            float[] kData = new float[seqLen * kvDim];
            float[] vData = new float[seqLen * kvDim];

            DenseTensor<float> q = new DenseTensor<float>(qData, new int[] { seqLen, qDim });
            DenseTensor<float> k = new DenseTensor<float>(kData, new int[] { seqLen, kvDim });
            DenseTensor<float> v = new DenseTensor<float>(vData, new int[] { seqLen, kvDim });

            OnnxTensorOperations.Linear(xNorm, _wq, null, q);
            OnnxTensorOperations.Linear(xNorm, _wk, null, k);
            OnnxTensorOperations.Linear(xNorm, _wv, null, v);

            int halfDim = HEAD_DIM / 2;
            int ropeSize = seqLen * halfDim; // Actually it's just halfDim if we reuse... no, wait.
            // Rope expects cos/sin for each sequence element if seqLen > 1?
            // In Encoder it precomputes. In Decoder, it computes inside Forward.
            // The method signature of ComputeRope in DecoderLayer computes for 'seqLen'.
            // So we need arrays of size [seqLen, halfDim] flattened.
            // The size is seqLen * halfDim.

            // Use stackalloc if small enough, else array.
            // Note: max seqLen during prefill can be large. But decoding is 1.
            // 4096 floats = 16KB. Stack is large enough.

            // Warning: seqLen * halfDim can be large for long sequences.
            // 1500 tokens * 64 = 96000 floats. Too big for stackalloc.
            // Use array pool or just array. The original code uses new float[] if > 4096.

            Span<float> cos = ropeSize <= 4096 ? stackalloc float[ropeSize] : new float[ropeSize];
            Span<float> sin = ropeSize <= 4096 ? stackalloc float[ropeSize] : new float[ropeSize];

            ComputeRope(pos, seqLen, cos, sin);

            OnnxTensorOperations.ApplyRoPE(q.Buffer.Span, cos, sin, seqLen, HEADS, HEAD_DIM);
            OnnxTensorOperations.ApplyRoPE(k.Buffer.Span, cos, sin, seqLen, KV_HEADS, HEAD_DIM);

            UpdateKVCache(k, v, pos, seqLen);

            float[] attnOutData = new float[seqLen * qDim];
            DenseTensor<float> attnOut = new DenseTensor<float>(attnOutData, new int[] { seqLen, qDim });

            PerformAttention(q, attnOut, pos, seqLen);

            float[] projOutData = new float[seqLen * DIM];
            DenseTensor<float> projOut = new DenseTensor<float>(projOutData, new int[] { seqLen, DIM });

            OnnxTensorOperations.Linear(attnOut, _wo, null, projOut);

            TensorPrimitives.Add(hSpan, projOut.Buffer.Span, hSpan);

            var ffnNormSpan = _ffnNorm.Buffer.Span;
            for (int s = 0; s < seqLen; s++)
            {
                OnnxTensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), ffnNormSpan, xNormSpan.Slice(s*DIM, DIM), NORM_EPS);
            }

            if (tCond != null)
            {
                float[] adaHiddenData = new float[32];
                DenseTensor<float> adaHidden = new DenseTensor<float>(adaHiddenData, new int[] { 32 });

                OnnxTensorOperations.Linear(tCond, _adaDown, null, adaHidden);
                OnnxTensorOperations.Gelu(adaHidden, adaHidden);

                float[] adaScaleData = new float[DIM];
                DenseTensor<float> adaScale = new DenseTensor<float>(adaScaleData, new int[] { DIM });

                OnnxTensorOperations.Linear(adaHidden, _adaUp, null, adaScale);

                var adaScaleSpan = adaScale.Buffer.Span;

                for (int s = 0; s < seqLen; s++)
                {
                    var xn = xNormSpan.Slice(s * DIM, DIM);
                    for (int d = 0; d < DIM; d++)
                    {
                        xn[d] *= (1.0f + adaScaleSpan[d]);
                    }
                }
            }

            float[] gateData = new float[seqLen * HIDDEN];
            float[] upData = new float[seqLen * HIDDEN];
            DenseTensor<float> gate = new DenseTensor<float>(gateData, new int[] { seqLen, HIDDEN });
            DenseTensor<float> up = new DenseTensor<float>(upData, new int[] { seqLen, HIDDEN });

            OnnxTensorOperations.Linear(xNorm, _w1, null, gate);
            OnnxTensorOperations.Linear(xNorm, _w3, null, up);

            OnnxTensorOperations.SiLU(gate, gate);
            TensorPrimitives.Multiply(gate.Buffer.Span, up.Buffer.Span, gate.Buffer.Span);

            OnnxTensorOperations.Linear(gate, _w2, null, projOut);

            TensorPrimitives.Add(hSpan, projOut.Buffer.Span, hSpan);
        }

        private void ComputeRope(int pos, int seqLen, Span<float> cos, Span<float> sin)
        {
            int halfDim = HEAD_DIM / 2;
            var ropeFreqsSpan = _ropeFreqs.Buffer.Span;

            for (int s = 0; s < seqLen; s++)
            {
                int absPos = pos + s;
                for (int i = 0; i < halfDim; i++)
                {
                    float angle = absPos * ropeFreqsSpan[i];
                    cos[s * halfDim + i] = MathF.Cos(angle);
                    sin[s * halfDim + i] = MathF.Sin(angle);
                }
            }
        }

        private void UpdateKVCache(DenseTensor<float> k, DenseTensor<float> v, int pos, int seqLen)
        {
            int dim = KV_HEADS * HEAD_DIM;
            var kSpan = k.Buffer.Span;
            var vSpan = v.Buffer.Span;
            var kCacheSpan = _kCache.Buffer.Span;
            var vCacheSpan = _vCache.Buffer.Span;

            for (int s = 0; s < seqLen; s++)
            {
                int absPos = pos + s;
                int cacheIdx = absPos % WINDOW;

                var srcK = kSpan.Slice(s * dim, dim);
                var dstK = kCacheSpan.Slice(cacheIdx * dim, dim);
                srcK.CopyTo(dstK);

                var srcV = vSpan.Slice(s * dim, dim);
                var dstV = vCacheSpan.Slice(cacheIdx * dim, dim);
                srcV.CopyTo(dstV);
            }
        }

        private unsafe void PerformAttention(DenseTensor<float> q, DenseTensor<float> output, int pos, int seqLen)
        {
            int qHeads = HEADS;
            int kvHeads = KV_HEADS;
            int gqaRatio = qHeads / kvHeads;
            int dim = HEAD_DIM;
            float scale = 1.0f / MathF.Sqrt(dim);

            fixed (float* pQ = q.Buffer.Span)
            fixed (float* pOut = output.Buffer.Span)
            fixed (float* pKCache = _kCache.Buffer.Span)
            fixed (float* pVCache = _vCache.Buffer.Span)
            {
                float* ptrQ = pQ;
                float* ptrOut = pOut;
                float* ptrKCache = pKCache;
                float* ptrVCache = pVCache;

                Parallel.For(0, qHeads, h =>
                {
                    int kvH = h / gqaRatio;

                    for (int s = 0; s < seqLen; s++)
                    {
                        int absPos = pos + s;

                        int qiOffset = (s * qHeads + h) * dim;
                        var qi = new ReadOnlySpan<float>(ptrQ + qiOffset, dim);

                        int startP = Math.Max(0, absPos - WINDOW + 1);
                        int endP = absPos;
                        int len = endP - startP + 1;

                        float[] scores = new float[len];

                        for (int j = 0; j < len; j++)
                        {
                            int pIdx = startP + j;
                            int cacheIdx = pIdx % WINDOW;
                            int kOffset = (cacheIdx * kvHeads + kvH) * dim;
                            var kj = new ReadOnlySpan<float>(ptrKCache + kOffset, dim);

                            scores[j] = TensorPrimitives.Dot(qi, kj) * scale;
                        }

                        OnnxTensorOperations.Softmax(scores);

                        int outOffset = (s * qHeads + h) * dim;
                        var outH = new Span<float>(ptrOut + outOffset, dim);
                        outH.Clear();

                        for (int j = 0; j < len; j++)
                        {
                            int pIdx = startP + j;
                            int cacheIdx = pIdx % WINDOW;
                            int vOffset = (cacheIdx * kvHeads + kvH) * dim;
                            var vj = new ReadOnlySpan<float>(ptrVCache + vOffset, dim);

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
}
