using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Voxtral
{
    public class Encoder
    {
        private const int LAYERS = 32;
        private const int DIM = 1280;
        private const int HEADS = 32;
        private const int HEAD_DIM = 64;
        private const int HIDDEN = 5120;
        private const float NORM_EPS = 1e-5f;
        private const float ROPE_THETA = 1000000.0f;

        private readonly ConvStem _convStem;
        private readonly EncoderLayer[] _layers;
        private readonly Tensor<float> _normWeight;

        public Encoder(SafetensorsReader reader)
        {
            _convStem = new ConvStem(reader);
            _layers = new EncoderLayer[LAYERS];
            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i] = new EncoderLayer(reader, i);
            }
            _normWeight = reader.LoadTensor("mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight");
        }

        public Tensor<float> Forward(Tensor<float> mel, out int seqLen)
        {
            var hArr = _convStem.Forward(mel, out seqLen);

            Tensor<float> cos, sin;
            ComputeRope(seqLen, out cos, out sin);

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(hArr, seqLen, cos, sin);
            }

            var hSpan = hArr.AsSpan();
            var normSpan = _normWeight.AsSpan();

            for (int s = 0; s < seqLen; s++)
            {
                var token = hSpan.Slice(s * DIM, DIM);
                TensorOperations.RMSNorm(token, normSpan, token, NORM_EPS);
            }

            return hArr;
        }

        private void ComputeRope(int seqLen, out Tensor<float> cos, out Tensor<float> sin)
        {
            int halfDim = HEAD_DIM / 2;
            float[] cosData = new float[seqLen * halfDim];
            float[] sinData = new float[seqLen * halfDim];

            for (int s = 0; s < seqLen; s++)
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float freq = 1.0f / MathF.Pow(ROPE_THETA, 2.0f * i / HEAD_DIM);
                    float angle = s * freq;
                    cosData[s * halfDim + i] = MathF.Cos(angle);
                    sinData[s * halfDim + i] = MathF.Sin(angle);
                }
            }

            cos = Tensor.Create(cosData, new nint[] { seqLen, halfDim });
            sin = Tensor.Create(sinData, new nint[] { seqLen, halfDim });
        }
    }

    class ConvStem
    {
        private readonly Tensor<float> _w0, _b0;
        private readonly Tensor<float> _w1, _b1;

        public ConvStem(SafetensorsReader reader)
        {
            string p = "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers";
            _w0 = reader.LoadTensor($"{p}.0.conv.weight");
            _b0 = reader.LoadTensor($"{p}.0.conv.bias");
            _w1 = reader.LoadTensor($"{p}.1.conv.weight");
            _b1 = reader.LoadTensor($"{p}.1.conv.bias");
        }

        public Tensor<float> Forward(Tensor<float> mel, out int outFrames)
        {
            TensorSpan<float> melSpan = mel;
            int frames = (int)melSpan.Lengths[1];

            float[] h0Data = new float[frames * 1280];
            Tensor<float> h0 = Tensor.Create(h0Data, new nint[] { 1280, frames });

            ApplyConv1dInterleaved(mel, _w0, _b0, h0, 128, 1280, 3, 1, frames);

            TensorOperations.Gelu(h0, h0);

            int frames2 = (frames + 1) / 2;

            float[] out1Data = new float[1280 * frames2];
            Tensor<float> out1 = Tensor.Create(out1Data, new nint[] { 1280, frames2 });

            ApplyConv1dInterleaved(h0, _w1, _b1, out1, 1280, 1280, 3, 2, frames);

            TensorOperations.Gelu(out1, out1);

            float[] outTData = new float[frames2 * 1280];
            var sOut1 = out1.AsSpan();

            for(int t=0; t<frames2; t++)
            {
                for(int c=0; c<1280; c++)
                {
                    outTData[t * 1280 + c] = sOut1[c * frames2 + t];
                }
            }

            Tensor<float> outT = Tensor.Create(outTData, new nint[] { frames2, 1280 });

            int trunc = frames2 % 4;
            if (trunc > 0)
            {
                int newLen = frames2 - trunc;
                float[] truncatedData = new float[newLen * 1280];
                Array.Copy(outTData, trunc * 1280, truncatedData, 0, newLen * 1280);

                outFrames = newLen;
                return Tensor.Create(truncatedData, new nint[] { newLen, 1280 });
            }

            outFrames = frames2;
            return outT;
        }

        private void ApplyConv1dInterleaved(Tensor<float> input, Tensor<float> weight, Tensor<float> bias, Tensor<float> output,
                                            int cIn, int cOut, int kSize, int stride, int len)
        {
            int paddingTotal = kSize - stride;
            int outLen = (len + paddingTotal - kSize) / stride + 1;

            Parallel.For(0, cOut, o =>
            {
                var inSpan = input.AsTensorSpan();
                var wSpan = weight.AsTensorSpan();
                var bSpan = bias.AsTensorSpan();
                var outSpan = output.AsTensorSpan();

                float b = bSpan[o];

                for (int t = 0; t < outLen; t++)
                {
                    float sum = b;

                    for (int k = 0; k < kSize; k++)
                    {
                        int inT = t * stride - paddingTotal + k;

                        if (inT >= 0 && inT < len)
                        {
                            for (int i = 0; i < cIn; i++)
                            {
                                nint inputIdx = (nint)i * len + inT;
                                nint weightIdx = (nint)o * cIn * kSize + (nint)i * kSize + k;

                                sum += inSpan[inputIdx] * wSpan[weightIdx];
                            }
                        }
                    }
                    outSpan[(nint)o * outLen + t] = sum;
                }
            });
        }
    }

    class EncoderLayer
    {
        private readonly Tensor<float> _attnNorm, _ffnNorm;
        private readonly Tensor<float> _wq, _wq_b;
        private readonly Tensor<float> _wk;
        private readonly Tensor<float> _wv, _wv_b;
        private readonly Tensor<float> _wo, _wo_b;
        private readonly Tensor<float> _w1, _w2, _w3, _w2_b;

        private const int DIM = 1280;
        private const int HEADS = 32;
        private const int HEAD_DIM = 64;
        private const int HIDDEN = 5120;
        private const float NORM_EPS = 1e-5f;

        public EncoderLayer(SafetensorsReader reader, int layerIdx)
        {
            string p = $"mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{layerIdx}";

            _attnNorm = reader.LoadTensor($"{p}.attention_norm.weight");
            _ffnNorm = reader.LoadTensor($"{p}.ffn_norm.weight");

            _wq = reader.LoadTensor($"{p}.attention.wq.weight");
            _wq_b = reader.LoadTensor($"{p}.attention.wq.bias");
            _wk = reader.LoadTensor($"{p}.attention.wk.weight");
            _wv = reader.LoadTensor($"{p}.attention.wv.weight");
            _wv_b = reader.LoadTensor($"{p}.attention.wv.bias");
            _wo = reader.LoadTensor($"{p}.attention.wo.weight");
            _wo_b = reader.LoadTensor($"{p}.attention.wo.bias");

            _w1 = reader.LoadTensor($"{p}.feed_forward.w1.weight");
            _w2 = reader.LoadTensor($"{p}.feed_forward.w2.weight");
            _w2_b = reader.LoadTensor($"{p}.feed_forward.w2.bias");
            _w3 = reader.LoadTensor($"{p}.feed_forward.w3.weight");
        }

        public void Forward(Tensor<float> h, int seqLen, Tensor<float> cos, Tensor<float> sin)
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

            float[] qData = new float[seqLen * qDim];
            float[] kData = new float[seqLen * qDim];
            float[] vData = new float[seqLen * qDim];

            Tensor<float> q = Tensor.Create(qData, new nint[] { seqLen, qDim });
            Tensor<float> k = Tensor.Create(kData, new nint[] { seqLen, qDim });
            Tensor<float> v = Tensor.Create(vData, new nint[] { seqLen, qDim });

            TensorOperations.Linear(xNorm, _wq, _wq_b, q);
            TensorOperations.Linear(xNorm, _wk, null, k);
            TensorOperations.Linear(xNorm, _wv, _wv_b, v);

            TensorOperations.ApplyRoPE(q, cos, sin, seqLen, HEADS, HEAD_DIM);
            TensorOperations.ApplyRoPE(k, cos, sin, seqLen, HEADS, HEAD_DIM);

            float[] attnOutData = new float[seqLen * qDim];
            Tensor<float> attnOut = Tensor.Create(attnOutData, new nint[] { seqLen, qDim });

            PerformAttention(q, k, v, attnOut, seqLen);

            float[] projOutData = new float[seqLen * DIM];
            Tensor<float> projOut = Tensor.Create(projOutData, new nint[] { seqLen, DIM });

            TensorOperations.Linear(attnOut, _wo, _wo_b, projOut);

            TensorPrimitives.Add(hSpan, projOut.AsSpan(), hSpan);

            var ffnNormSpan = _ffnNorm.AsSpan();
            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), ffnNormSpan, xNormSpan.Slice(s*DIM, DIM), NORM_EPS);
            }

            float[] gateData = new float[seqLen * HIDDEN];
            float[] upData = new float[seqLen * HIDDEN];
            Tensor<float> gate = Tensor.Create(gateData, new nint[] { seqLen, HIDDEN });
            Tensor<float> up = Tensor.Create(upData, new nint[] { seqLen, HIDDEN });

            TensorOperations.Linear(xNorm, _w1, null, gate);
            TensorOperations.Linear(xNorm, _w3, null, up);

            TensorOperations.SiLU(gate, gate);
            TensorPrimitives.Multiply(gate.AsSpan(), up.AsSpan(), gate.AsSpan());

            TensorOperations.Linear(gate, _w2, _w2_b, projOut);

            TensorPrimitives.Add(hSpan, projOut.AsSpan(), hSpan);
        }

        private void PerformAttention(Tensor<float> q, Tensor<float> k, Tensor<float> v, Tensor<float> output, int seqLen)
        {
            int window = 750;
            float scale = 1.0f / MathF.Sqrt(HEAD_DIM);

            Parallel.For(0, HEADS, h =>
            {
                var qSpan = q.AsTensorSpan();
                var kSpan = k.AsTensorSpan();
                var vSpan = v.AsTensorSpan();
                var outSpan = output.AsTensorSpan();

                for (int i = 0; i < seqLen; i++)
                {
                    nint qiOffset = ((nint)i * HEADS + h) * HEAD_DIM;
                    var qi = qSpan.Slice(qiOffset, HEAD_DIM);

                    int startJ = Math.Max(0, i - window + 1);
                    int endJ = i;
                    int len = endJ - startJ + 1;

                    // Use ArrayPool to avoid allocations and stack overflow
                    float[] scoresArray = ArrayPool<float>.Shared.Rent(len);
                    Span<float> scores = scoresArray.AsSpan(0, len);

                    try
                    {
                        for (int j = 0; j < len; j++)
                        {
                            int realJ = startJ + j;
                            nint kjOffset = ((nint)realJ * HEADS + h) * HEAD_DIM;
                            var kj = kSpan.Slice(kjOffset, HEAD_DIM);
                            scores[j] = Tensor.Dot<float>(qi, kj) * scale;
                        }

                        TensorOperations.Softmax(scores);

                        nint outOffset = ((nint)i * HEADS + h) * HEAD_DIM;
                        var outH = outSpan.Slice(outOffset, HEAD_DIM);
                        outH.Fill(0);

                        for (int j = 0; j < len; j++)
                        {
                            int realJ = startJ + j;
                            nint vjOffset = ((nint)realJ * HEADS + h) * HEAD_DIM;
                            var vj = vSpan.Slice(vjOffset, HEAD_DIM);
                            float score = scores[j];

                            for (int d = 0; d < HEAD_DIM; d++)
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
