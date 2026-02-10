using System;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

namespace Voxtral.Onnx
{
    public class OnnxEncoder
    {
        private const int LAYERS = 32;
        private const int DIM = 1280;
        private const int HEADS = 32;
        private const int HEAD_DIM = 64;
        private const int HIDDEN = 5120;
        private const float NORM_EPS = 1e-5f;
        private const float ROPE_THETA = 1000000.0f;

        private readonly OnnxConvStem _convStem;
        private readonly OnnxEncoderLayer[] _layers;
        private readonly DenseTensor<float> _normWeight;

        public OnnxEncoder(OnnxSafetensorsReader reader)
        {
            _convStem = new OnnxConvStem(reader);
            _layers = new OnnxEncoderLayer[LAYERS];
            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i] = new OnnxEncoderLayer(reader, i);
            }
            _normWeight = reader.LoadTensor("mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight");
        }

        public DenseTensor<float> Forward(DenseTensor<float> mel, out int seqLen)
        {
            var hArr = _convStem.Forward(mel, out seqLen);

            DenseTensor<float> cos, sin;
            ComputeRope(seqLen, out cos, out sin);

            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(hArr, seqLen, cos, sin);
            }

            var hSpan = hArr.Buffer.Span;
            var normSpan = _normWeight.Buffer.Span;

            for (int s = 0; s < seqLen; s++)
            {
                var token = hSpan.Slice(s * DIM, DIM);
                OnnxTensorOperations.RMSNorm(token, normSpan, token, NORM_EPS);
            }

            return hArr;
        }

        private void ComputeRope(int seqLen, out DenseTensor<float> cos, out DenseTensor<float> sin)
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

            cos = new DenseTensor<float>(cosData, new int[] { seqLen, halfDim });
            sin = new DenseTensor<float>(sinData, new int[] { seqLen, halfDim });
        }
    }

    class OnnxConvStem
    {
        private readonly DenseTensor<float> _w0, _b0;
        private readonly DenseTensor<float> _w1, _b1;

        public OnnxConvStem(OnnxSafetensorsReader reader)
        {
            string p = "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers";
            _w0 = reader.LoadTensor($"{p}.0.conv.weight");
            _b0 = reader.LoadTensor($"{p}.0.conv.bias");
            _w1 = reader.LoadTensor($"{p}.1.conv.weight");
            _b1 = reader.LoadTensor($"{p}.1.conv.bias");
        }

        public DenseTensor<float> Forward(DenseTensor<float> mel, out int outFrames)
        {
            int frames = mel.Dimensions[1];

            float[] h0Data = new float[frames * 1280];
            DenseTensor<float> h0 = new DenseTensor<float>(h0Data, new int[] { 1280, frames });

            ApplyConv1dInterleaved(mel.Buffer.Span, _w0.Buffer.Span, _b0.Buffer.Span, h0.Buffer.Span, 128, 1280, 3, 1, frames);

            OnnxTensorOperations.Gelu(h0, h0);

            int frames2 = (frames + 1) / 2;

            float[] out1Data = new float[1280 * frames2];
            DenseTensor<float> out1 = new DenseTensor<float>(out1Data, new int[] { 1280, frames2 });

            ApplyConv1dInterleaved(h0.Buffer.Span, _w1.Buffer.Span, _b1.Buffer.Span, out1.Buffer.Span, 1280, 1280, 3, 2, frames);

            OnnxTensorOperations.Gelu(out1, out1);

            float[] outTData = new float[frames2 * 1280];
            var sOut1 = out1.Buffer.Span;

            // Transpose
            for(int t=0; t<frames2; t++)
            {
                for(int c=0; c<1280; c++)
                {
                    outTData[t * 1280 + c] = sOut1[c * frames2 + t];
                }
            }

            int trunc = frames2 % 4;
            if (trunc > 0)
            {
                int newLen = frames2 - trunc;
                float[] truncatedData = new float[newLen * 1280];
                Array.Copy(outTData, trunc * 1280, truncatedData, 0, newLen * 1280);

                outFrames = newLen;
                return new DenseTensor<float>(truncatedData, new int[] { newLen, 1280 });
            }

            outFrames = frames2;
            return new DenseTensor<float>(outTData, new int[] { frames2, 1280 });
        }

        private unsafe void ApplyConv1dInterleaved(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, Span<float> output,
                                            int cIn, int cOut, int kSize, int stride, int len)
        {
            int paddingTotal = kSize - stride;
            int outLen = (len + paddingTotal - kSize) / stride + 1;

            fixed (float* pInput = input)
            fixed (float* pWeight = weight)
            fixed (float* pBias = bias)
            fixed (float* pOutput = output)
            {
                float* pIn = pInput;
                float* pW = pWeight;
                float* pB = pBias;
                float* pOut = pOutput;

                Parallel.For(0, cOut, o =>
                {
                    float b = pB[o];

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
                                    sum += pIn[i * len + inT] * pW[o * cIn * kSize + i * kSize + k];
                                }
                            }
                        }
                        pOut[o * outLen + t] = sum;
                    }
                });
            }
        }
    }

    class OnnxEncoderLayer
    {
        private readonly DenseTensor<float> _attnNorm, _ffnNorm;
        private readonly DenseTensor<float> _wq, _wq_b;
        private readonly DenseTensor<float> _wk;
        private readonly DenseTensor<float> _wv, _wv_b;
        private readonly DenseTensor<float> _wo, _wo_b;
        private readonly DenseTensor<float> _w1, _w2, _w3, _w2_b;

        private const int DIM = 1280;
        private const int HEADS = 32;
        private const int HEAD_DIM = 64;
        private const int HIDDEN = 5120;
        private const float NORM_EPS = 1e-5f;

        public OnnxEncoderLayer(OnnxSafetensorsReader reader, int layerIdx)
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

        public void Forward(DenseTensor<float> h, int seqLen, DenseTensor<float> cos, DenseTensor<float> sin)
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

            float[] qData = new float[seqLen * qDim];
            float[] kData = new float[seqLen * qDim];
            float[] vData = new float[seqLen * qDim];

            DenseTensor<float> q = new DenseTensor<float>(qData, new int[] { seqLen, qDim });
            DenseTensor<float> k = new DenseTensor<float>(kData, new int[] { seqLen, qDim });
            DenseTensor<float> v = new DenseTensor<float>(vData, new int[] { seqLen, qDim });

            OnnxTensorOperations.Linear(xNorm, _wq, _wq_b, q);
            OnnxTensorOperations.Linear(xNorm, _wk, null, k);
            OnnxTensorOperations.Linear(xNorm, _wv, _wv_b, v);

            OnnxTensorOperations.ApplyRoPE(q, cos, sin, seqLen, HEADS, HEAD_DIM);
            OnnxTensorOperations.ApplyRoPE(k, cos, sin, seqLen, HEADS, HEAD_DIM);

            float[] attnOutData = new float[seqLen * qDim];
            DenseTensor<float> attnOut = new DenseTensor<float>(attnOutData, new int[] { seqLen, qDim });

            PerformAttention(q, k, v, attnOut, seqLen);

            float[] projOutData = new float[seqLen * DIM];
            DenseTensor<float> projOut = new DenseTensor<float>(projOutData, new int[] { seqLen, DIM });

            OnnxTensorOperations.Linear(attnOut, _wo, _wo_b, projOut);

            TensorPrimitives.Add(hSpan, projOut.Buffer.Span, hSpan);

            var ffnNormSpan = _ffnNorm.Buffer.Span;
            for (int s = 0; s < seqLen; s++)
            {
                OnnxTensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), ffnNormSpan, xNormSpan.Slice(s*DIM, DIM), NORM_EPS);
            }

            float[] gateData = new float[seqLen * HIDDEN];
            float[] upData = new float[seqLen * HIDDEN];
            DenseTensor<float> gate = new DenseTensor<float>(gateData, new int[] { seqLen, HIDDEN });
            DenseTensor<float> up = new DenseTensor<float>(upData, new int[] { seqLen, HIDDEN });

            OnnxTensorOperations.Linear(xNorm, _w1, null, gate);
            OnnxTensorOperations.Linear(xNorm, _w3, null, up);

            OnnxTensorOperations.SiLU(gate, gate);
            TensorPrimitives.Multiply(gate.Buffer.Span, up.Buffer.Span, gate.Buffer.Span);

            OnnxTensorOperations.Linear(gate, _w2, _w2_b, projOut);

            TensorPrimitives.Add(hSpan, projOut.Buffer.Span, hSpan);
        }

        private unsafe void PerformAttention(DenseTensor<float> q, DenseTensor<float> k, DenseTensor<float> v, DenseTensor<float> output, int seqLen)
        {
            int window = 750;
            float scale = 1.0f / MathF.Sqrt(HEAD_DIM);

            fixed (float* pQ = q.Buffer.Span)
            fixed (float* pK = k.Buffer.Span)
            fixed (float* pV = v.Buffer.Span)
            fixed (float* pOut = output.Buffer.Span)
            {
                float* ptrQ = pQ;
                float* ptrK = pK;
                float* ptrV = pV;
                float* ptrOut = pOut;

                Parallel.For(0, HEADS, h =>
                {
                    for (int i = 0; i < seqLen; i++)
                    {
                        int qiOffset = (i * HEADS + h) * HEAD_DIM;
                        var qi = new ReadOnlySpan<float>(ptrQ + qiOffset, HEAD_DIM);

                        int startJ = Math.Max(0, i - window + 1);
                        int endJ = i;
                        int len = endJ - startJ + 1;

                        float[] scores = new float[len];

                        for (int j = 0; j < len; j++)
                        {
                            int realJ = startJ + j;
                            int kjOffset = (realJ * HEADS + h) * HEAD_DIM;
                            var kj = new ReadOnlySpan<float>(ptrK + kjOffset, HEAD_DIM);
                            scores[j] = TensorPrimitives.Dot(qi, kj) * scale;
                        }

                        OnnxTensorOperations.Softmax(scores);

                        int outOffset = (i * HEADS + h) * HEAD_DIM;
                        var outH = new Span<float>(ptrOut + outOffset, HEAD_DIM);
                        outH.Clear();

                        for (int j = 0; j < len; j++)
                        {
                            int realJ = startJ + j;
                            int vjOffset = (realJ * HEADS + h) * HEAD_DIM;
                            var vj = new ReadOnlySpan<float>(ptrV + vjOffset, HEAD_DIM);
                            float score = scores[j];

                            for (int d = 0; d < HEAD_DIM; d++)
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
