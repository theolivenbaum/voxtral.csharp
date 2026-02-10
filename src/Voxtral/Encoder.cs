using System;
using System.Numerics.Tensors;

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

        public float[] Forward(Tensor<float> mel, out int seqLen)
        {
            // mel: [128, frames]
            // ConvStem
            var hArr = _convStem.Forward(mel, out seqLen);

            // hArr: [seqLen * 1280]

            // RoPE
            float[] cos, sin;
            ComputeRope(seqLen, out cos, out sin);

            // Layers
            for (int i = 0; i < LAYERS; i++)
            {
                _layers[i].Forward(hArr, seqLen, cos, sin);
            }

            // Final Norm
            var hSpan = hArr.AsSpan();
            for (int s = 0; s < seqLen; s++)
            {
                var token = hSpan.Slice(s * DIM, DIM);
                TensorOperations.RMSNorm(token, _normWeight.AsReadOnlyTensorSpan(), token, NORM_EPS);
            }

            return hArr;
        }

        private void ComputeRope(int seqLen, out float[] cos, out float[] sin)
        {
            int halfDim = HEAD_DIM / 2;
            cos = new float[seqLen * halfDim];
            sin = new float[seqLen * halfDim];

            for (int s = 0; s < seqLen; s++)
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float freq = 1.0f / MathF.Pow(ROPE_THETA, 2.0f * i / HEAD_DIM);
                    float angle = s * freq;
                    cos[s * halfDim + i] = MathF.Cos(angle);
                    sin[s * halfDim + i] = MathF.Sin(angle);
                }
            }
        }
    }

    class ConvStem
    {
        private readonly Tensor<float> _w0, _b0;
        private readonly Tensor<float> _w1, _b1;

        public ConvStem(SafetensorsReader reader)
        {
            string p = "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers";
            _w0 = reader.LoadTensor($"{p}.0.conv.weight"); // [1280, 128, 3]
            _b0 = reader.LoadTensor($"{p}.0.conv.bias");   // [1280]
            _w1 = reader.LoadTensor($"{p}.1.conv.weight"); // [1280, 1280, 3]
            _b1 = reader.LoadTensor($"{p}.1.conv.bias");   // [1280]
        }

        public float[] Forward(Tensor<float> mel, out int outFrames)
        {
            // mel: [128, frames]
            int frames = (int)mel.Lengths[1];

            // First conv
            float[] h0 = new float[frames * 1280];
            // Reuse input array if possible? No, float[].

            ApplyConv1dInterleaved(mel.AsReadOnlyTensorSpan(), _w0.AsReadOnlyTensorSpan(), _b0.AsReadOnlyTensorSpan(), h0, 128, 1280, 3, 1, frames);

            // GELU
            TensorOperations.Gelu(h0, h0);

            // Second conv
            // Output frames calculation
            int frames2 = (frames + 1) / 2; // Stride 2

            float[] out1 = new float[1280 * frames2];
            ApplyConv1dInterleaved(h0.AsSpan(), _w1.AsReadOnlyTensorSpan(), _b1.AsReadOnlyTensorSpan(), out1, 1280, 1280, 3, 2, frames);

            // GELU
            TensorOperations.Gelu(out1, out1);

            // Transpose to [frames, 1280]
            float[] outT = new float[frames2 * 1280];
            for(int t=0; t<frames2; t++)
            {
                for(int c=0; c<1280; c++)
                {
                    outT[t * 1280 + c] = out1[c * frames2 + t];
                }
            }

            // Truncate
            int trunc = frames2 % 4;
            if (trunc > 0)
            {
                int newLen = frames2 - trunc;
                float[] truncated = new float[newLen * 1280];
                Array.Copy(outT, trunc * 1280, truncated, 0, newLen * 1280);
                outFrames = newLen;
                return truncated;
            }

            outFrames = frames2;
            return outT;
        }

        private void ApplyConv1dInterleaved(ReadOnlyTensorSpan<float> input, ReadOnlyTensorSpan<float> weight, ReadOnlyTensorSpan<float> bias, Span<float> output,
                                            int cIn, int cOut, int kSize, int stride, int len)
        {
            unsafe
            {
                fixed (float* pIn = input)
                fixed (float* pW = weight)
                fixed (float* pB = bias)
                {
                     ApplyConv1dInterleavedImpl(new ReadOnlySpan<float>(pIn, (int)input.FlattenedLength),
                                                new ReadOnlySpan<float>(pW, (int)weight.FlattenedLength),
                                                new ReadOnlySpan<float>(pB, (int)bias.FlattenedLength),
                                                output, cIn, cOut, kSize, stride, len);
                }
            }
        }

        private void ApplyConv1dInterleaved(ReadOnlySpan<float> input, ReadOnlyTensorSpan<float> weight, ReadOnlyTensorSpan<float> bias, Span<float> output,
                                            int cIn, int cOut, int kSize, int stride, int len)
        {
            unsafe
            {
                fixed (float* pW = weight)
                fixed (float* pB = bias)
                {
                     ApplyConv1dInterleavedImpl(input,
                                                new ReadOnlySpan<float>(pW, (int)weight.FlattenedLength),
                                                new ReadOnlySpan<float>(pB, (int)bias.FlattenedLength),
                                                output, cIn, cOut, kSize, stride, len);
                }
            }
        }

        private void ApplyConv1dInterleavedImpl(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, Span<float> output,
                                            int cIn, int cOut, int kSize, int stride, int len)
        {
            int pad = (kSize - 1) / 2; // symmetric padding? No, wait.
            // Python causal_conv1d pads LEFT only.
            // padding_total = kSize - stride.
            int paddingTotal = kSize - stride;

            int outLen = (len + paddingTotal - kSize) / stride + 1;
            // Wait, standard conv output length formula: floor((L + 2*pad - dilation*(k-1) - 1)/stride + 1)
            // Python:
            // n_frames = (x.shape[-1] - kSize + paddingTotal) / stride + 1
            // This is equivalent to floor((L + P - K)/S + 1).

            // Note: in my previous manual calculation I used a different formula.
            // Let's trust Python formula logic.

            // Parallelize over cOut
            // Note: System.Threading.Tasks.Parallel.For is good here.

            // For now simple loop
            for (int o = 0; o < cOut; o++)
            {
                float b = bias[o];

                for (int t = 0; t < outLen; t++)
                {
                    float sum = b;

                    for (int k = 0; k < kSize; k++)
                    {
                        // input index: t * stride - paddingTotal + k
                        int inT = t * stride - paddingTotal + k;

                        if (inT >= 0 && inT < len)
                        {
                            for (int i = 0; i < cIn; i++)
                            {
                                // weight index: o * cIn * kSize + i * kSize + k
                                // input index: i * len + inT
                                sum += input[i * len + inT] * weight[o * cIn * kSize + i * kSize + k];
                            }
                        }
                    }
                    output[o * outLen + t] = sum;
                }
            }
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

        public void Forward(float[] h, int seqLen, float[] cos, float[] sin)
        {
            // h: [seqLen * 1280]
            var hSpan = h.AsSpan();

            float[] xNorm = new float[seqLen * DIM];

            // 1. RMSNorm
            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), _attnNorm.AsReadOnlyTensorSpan(), xNorm.AsSpan(s*DIM, DIM), NORM_EPS);
            }

            // 2. Attention
            int qDim = HEADS * HEAD_DIM; // 2048
            float[] q = new float[seqLen * qDim];
            float[] k = new float[seqLen * qDim];
            float[] v = new float[seqLen * qDim];

            TensorOperations.Linear(xNorm, _wq.AsReadOnlyTensorSpan(), _wq_b.AsReadOnlyTensorSpan(), q, seqLen, qDim, DIM);
            TensorOperations.Linear(xNorm, _wk.AsReadOnlyTensorSpan(), ReadOnlyTensorSpan<float>.Empty, k, seqLen, qDim, DIM);
            TensorOperations.Linear(xNorm, _wv.AsReadOnlyTensorSpan(), _wv_b.AsReadOnlyTensorSpan(), v, seqLen, qDim, DIM);

            // RoPE
            TensorOperations.ApplyRoPE(q, cos, sin, seqLen, HEADS, HEAD_DIM);
            TensorOperations.ApplyRoPE(k, cos, sin, seqLen, HEADS, HEAD_DIM);

            // Self-Attention
            float[] attnOut = new float[seqLen * qDim];
            PerformAttention(q, k, v, attnOut, seqLen);

            // Output Projection
            float[] projOut = new float[seqLen * DIM];
            TensorOperations.Linear(attnOut, _wo.AsReadOnlyTensorSpan(), _wo_b.AsReadOnlyTensorSpan(), projOut, seqLen, DIM, qDim);

            TensorPrimitives.Add(hSpan, projOut, hSpan);

            // 3. FFN
            for (int s = 0; s < seqLen; s++)
            {
                TensorOperations.RMSNorm(hSpan.Slice(s*DIM, DIM), _ffnNorm.AsReadOnlyTensorSpan(), xNorm.AsSpan(s*DIM, DIM), NORM_EPS);
            }

            float[] gate = new float[seqLen * HIDDEN];
            float[] up = new float[seqLen * HIDDEN];

            TensorOperations.Linear(xNorm, _w1.AsReadOnlyTensorSpan(), ReadOnlyTensorSpan<float>.Empty, gate, seqLen, HIDDEN, DIM);
            TensorOperations.Linear(xNorm, _w3.AsReadOnlyTensorSpan(), ReadOnlyTensorSpan<float>.Empty, up, seqLen, HIDDEN, DIM);

            TensorOperations.SiLU(gate, gate);
            TensorPrimitives.Multiply(gate, up, gate);

            TensorOperations.Linear(gate, _w2.AsReadOnlyTensorSpan(), _w2_b.AsReadOnlyTensorSpan(), projOut, seqLen, DIM, HIDDEN);

            TensorPrimitives.Add(hSpan, projOut, hSpan);
        }

        private void PerformAttention(float[] q, float[] k, float[] v, float[] output, int seqLen)
        {
            int window = 750;
            float scale = 1.0f / MathF.Sqrt(HEAD_DIM);

            // Reusing buffer for scores to avoid allocs?
            // Need thread-local or stackalloc if small.
            // HEAD_DIM is 64. seqLen can be large.
            // window is 750.
            // So scores size is at most 751?

            // For now allocate inside. Optimized version would use buffer.

            for (int h = 0; h < HEADS; h++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    var qi = q.AsSpan((i * HEADS + h) * HEAD_DIM, HEAD_DIM);

                    int startJ = Math.Max(0, i - window + 1);
                    int endJ = i;
                    int len = endJ - startJ + 1;

                    // Allocating 750 floats is okayish.
                    float[] scores = new float[len];

                    for (int j = 0; j < len; j++)
                    {
                        int realJ = startJ + j;
                        var kj = k.AsSpan((realJ * HEADS + h) * HEAD_DIM, HEAD_DIM);
                        scores[j] = TensorPrimitives.Dot(qi, kj) * scale;
                    }

                    TensorOperations.Softmax(scores);

                    var outH = output.AsSpan((i * HEADS + h) * HEAD_DIM, HEAD_DIM);
                    outH.Clear();

                    for (int j = 0; j < len; j++)
                    {
                        int realJ = startJ + j;
                        var vj = v.AsSpan((realJ * HEADS + h) * HEAD_DIM, HEAD_DIM);
                        float score = scores[j];

                        // out += score * v
                        // Vectorized add with scalar mul
                        // No TensorPrimitives op for Add(dest, src * scalar).
                        // Loop manual
                        for (int d = 0; d < HEAD_DIM; d++)
                        {
                            outH[d] += score * vj[d];
                        }
                    }
                }
            }
        }
    }
}
