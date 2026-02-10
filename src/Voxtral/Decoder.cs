using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Threading.Tasks;

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

        private readonly Tensor<float> _tokEmbeddings; // [VOCAB, DIM]
        private readonly Tensor<float> _finalNorm;
        private readonly DecoderLayer[] _layers;

        public Decoder(SafetensorsReader reader)
        {
            _tokEmbeddings = reader.LoadTensor("mm_streams_embeddings.embedding_module.tok_embeddings.weight");
            _finalNorm = reader.LoadTensor("norm.weight");

            _layers = new DecoderLayer[LAYERS];
            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i] = new DecoderLayer(reader, i);
            }
        }

        public Tensor<float> EmbedToken(int tokenId)
        {
            float[] embedData = new float[DIM];
            Tensor<float> embed = Tensor.Create(embedData, new nint[] { DIM });

            var src = _tokEmbeddings.AsSpan().Slice(tokenId * DIM, DIM);
            src.CopyTo(embed.AsSpan());

            return embed;
        }

        public void Prefill(Tensor<float> inputEmbeds, Tensor<float> tCond)
        {
            int seqLen = (int)(inputEmbeds.AsSpan().Length / DIM);

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(inputEmbeds, 0, seqLen, tCond);
            }
        }

        public Tensor<float> ForwardOne(Tensor<float> embed, int pos, Tensor<float> tCond)
        {
            float[] hData = new float[DIM];
            Tensor<float> h = Tensor.Create(hData, new nint[] { DIM });
            embed.AsSpan().CopyTo(h.AsSpan());

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(h, pos, 1, tCond);
            }

            TensorOperations.RMSNorm(h, _finalNorm, h, NORM_EPS);

            float[] logitsData = new float[VOCAB_SIZE];
            Tensor<float> logits = Tensor.Create(logitsData, new nint[] { VOCAB_SIZE });

            TensorOperations.Linear(h, _tokEmbeddings, null, logits);

            return logits;
        }

        public static Tensor<float> ComputeTimeEmbedding(float t, int dim)
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
            return Tensor.Create(embData, new nint[] { dim });
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

        private readonly Tensor<float> _ropeFreqs;
        private readonly Tensor<float> _attnNorm, _ffnNorm;
        private readonly Tensor<float> _wq, _wk, _wv, _wo;
        private readonly Tensor<float> _w1, _w2, _w3;
        private readonly Tensor<float> _adaDown, _adaUp;

        private readonly Tensor<float> _kCache; // [WINDOW, KV_HEADS, HEAD_DIM]
        private readonly Tensor<float> _vCache; // [WINDOW, KV_HEADS, HEAD_DIM]

        public DecoderLayer(SafetensorsReader reader, int layerIdx)
        {
            int halfDim = HEAD_DIM / 2;
            float[] ropeFreqsData = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                ropeFreqsData[i] = 1.0f / MathF.Pow(ROPE_THETA, 2.0f * i / HEAD_DIM);
            }
            _ropeFreqs = Tensor.Create(ropeFreqsData, new nint[] { halfDim });

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
            _kCache = Tensor.Create(kCacheData, new nint[] { WINDOW, KV_HEADS, HEAD_DIM });
            _vCache = Tensor.Create(vCacheData, new nint[] { WINDOW, KV_HEADS, HEAD_DIM });
        }

        public void Forward(Tensor<float> h, int pos, int seqLen, Tensor<float> tCond)
        {
            var hSpan = h.AsSpan();
            float[] xNormData = new float[seqLen * DIM];
            Tensor<float> xNorm = Tensor.Create(xNormData, new nint[] { seqLen, DIM });
            var xNormSpan = xNorm.AsSpan();
            var attnNormSpan = _attnNorm.AsSpan();

            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), attnNormSpan, xNormSpan.Slice(s*DIM, DIM), NORM_EPS);
            }

            int qDim = HEADS * HEAD_DIM;
            int kvDim = KV_HEADS * HEAD_DIM;

            float[] qData = new float[seqLen * qDim];
            float[] kData = new float[seqLen * kvDim];
            float[] vData = new float[seqLen * kvDim];

            Tensor<float> q = Tensor.Create(qData, new nint[] { seqLen, qDim });
            Tensor<float> k = Tensor.Create(kData, new nint[] { seqLen, kvDim });
            Tensor<float> v = Tensor.Create(vData, new nint[] { seqLen, kvDim });

            TensorOperations.Linear(xNorm, _wq, null, q);
            TensorOperations.Linear(xNorm, _wk, null, k);
            TensorOperations.Linear(xNorm, _wv, null, v);

            int halfDim = HEAD_DIM / 2;
            int ropeSize = seqLen * halfDim;

            Span<float> cos = ropeSize <= 4096 ? stackalloc float[ropeSize] : new float[ropeSize];
            Span<float> sin = ropeSize <= 4096 ? stackalloc float[ropeSize] : new float[ropeSize];

            ComputeRope(pos, seqLen, cos, sin);

            TensorOperations.ApplyRoPE(q.AsSpan(), cos, sin, seqLen, HEADS, HEAD_DIM);
            TensorOperations.ApplyRoPE(k.AsSpan(), cos, sin, seqLen, KV_HEADS, HEAD_DIM);

            UpdateKVCache(k, v, pos, seqLen);

            float[] attnOutData = new float[seqLen * qDim];
            Tensor<float> attnOut = Tensor.Create(attnOutData, new nint[] { seqLen, qDim });

            PerformAttention(q, attnOut, pos, seqLen);

            float[] projOutData = new float[seqLen * DIM];
            Tensor<float> projOut = Tensor.Create(projOutData, new nint[] { seqLen, DIM });

            TensorOperations.Linear(attnOut, _wo, null, projOut);

            TensorPrimitives.Add(hSpan, projOut.AsSpan(), hSpan);

            var ffnNormSpan = _ffnNorm.AsSpan();
            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), ffnNormSpan, xNormSpan.Slice(s*DIM, DIM), NORM_EPS);
            }

            if (tCond != null)
            {
                float[] adaHiddenData = new float[32];
                Tensor<float> adaHidden = Tensor.Create(adaHiddenData, new nint[] { 32 });

                TensorOperations.Linear(tCond, _adaDown, null, adaHidden);
                TensorOperations.Gelu(adaHidden, adaHidden);

                float[] adaScaleData = new float[DIM];
                Tensor<float> adaScale = Tensor.Create(adaScaleData, new nint[] { DIM });

                TensorOperations.Linear(adaHidden, _adaUp, null, adaScale);

                var adaScaleSpan = adaScale.AsSpan();

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
            Tensor<float> gate = Tensor.Create(gateData, new nint[] { seqLen, HIDDEN });
            Tensor<float> up = Tensor.Create(upData, new nint[] { seqLen, HIDDEN });

            TensorOperations.Linear(xNorm, _w1, null, gate);
            TensorOperations.Linear(xNorm, _w3, null, up);

            TensorOperations.SiLU(gate, gate);
            TensorPrimitives.Multiply(gate.AsSpan(), up.AsSpan(), gate.AsSpan());

            TensorOperations.Linear(gate, _w2, null, projOut);

            TensorPrimitives.Add(hSpan, projOut.AsSpan(), hSpan);
        }

        private void ComputeRope(int pos, int seqLen, Span<float> cos, Span<float> sin)
        {
            int halfDim = HEAD_DIM / 2;
            var ropeFreqsSpan = _ropeFreqs.AsSpan();

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

        private void UpdateKVCache(Tensor<float> k, Tensor<float> v, int pos, int seqLen)
        {
            int dim = KV_HEADS * HEAD_DIM;
            var kSpan = k.AsSpan();
            var vSpan = v.AsSpan();
            var kCacheSpan = _kCache.AsSpan();
            var vCacheSpan = _vCache.AsSpan();

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

        private void PerformAttention(Tensor<float> q, Tensor<float> output, int pos, int seqLen)
        {
            int qHeads = HEADS;
            int kvHeads = KV_HEADS;
            int gqaRatio = qHeads / kvHeads;
            int dim = HEAD_DIM;
            float scale = 1.0f / MathF.Sqrt(dim);

            Parallel.For(0, qHeads, h =>
            {
                var qSpan = q.AsSpan();
                var outSpan = output.AsSpan();
                var kCacheSpan = _kCache.AsSpan();
                var vCacheSpan = _vCache.AsSpan();

                int kvH = h / gqaRatio;

                for (int s = 0; s < seqLen; s++)
                {
                    int absPos = pos + s;

                    nint qiOffset = ((nint)s * qHeads + h) * dim;
                    var qi = qSpan.Slice((int)qiOffset, dim);
                    TensorSpan<float> qiTs = new TensorSpan<float>(qi);

                    int startP = Math.Max(0, absPos - WINDOW + 1);
                    int endP = absPos;
                    int len = endP - startP + 1;

                    // Use ArrayPool to avoid allocation overhead
                    float[] scoresArray = ArrayPool<float>.Shared.Rent(len);
                    Span<float> scores = scoresArray.AsSpan(0, len);

                    try
                    {
                        for (int j = 0; j < len; j++)
                        {
                            int pIdx = startP + j;
                            int cacheIdx = pIdx % WINDOW;
                            nint kOffset = ((nint)cacheIdx * kvHeads + kvH) * dim;
                            var kj = kCacheSpan.Slice((int)kOffset, dim);
                            TensorSpan<float> kjTs = new TensorSpan<float>(kj);

                            scores[j] = Tensor.Dot<float>(qiTs, kjTs) * scale;
                        }

                        TensorOperations.Softmax(scores);

                        nint outOffset = ((nint)s * qHeads + h) * dim;
                        var outH = outSpan.Slice((int)outOffset, dim);
                        outH.Fill(0);

                        for (int j = 0; j < len; j++)
                        {
                            int pIdx = startP + j;
                            int cacheIdx = pIdx % WINDOW;
                            nint vOffset = ((nint)cacheIdx * kvHeads + kvH) * dim;
                            var vj = vCacheSpan.Slice((int)vOffset, dim);

                            float score = scores[j];

                            for (int d = 0; d < dim; d++)
                            {
                                outH[d] += score * vj[d];
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(scoresArray);
                    }
                }
            });
        }
    }
}
