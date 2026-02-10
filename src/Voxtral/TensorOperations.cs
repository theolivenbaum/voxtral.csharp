using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Voxtral
{
    public static class TensorOperations
    {
        public static void RMSNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> y, float eps)
        {
            // x: [dim], w: [dim], y: [dim]
            float sumSq = TensorPrimitives.SumOfSquares(x);
            float rms = 1.0f / MathF.Sqrt(sumSq / x.Length + eps);

            TensorPrimitives.Multiply(x, w, y);
            TensorPrimitives.Multiply(y, rms, y);
        }

        public static void Softmax(Span<float> x)
        {
            float max = TensorPrimitives.Max(x);
            float sum = 0;

            for(int i=0; i<x.Length; i++)
            {
                float val = MathF.Exp(x[i] - max);
                x[i] = val;
                sum += val;
            }

            TensorPrimitives.Divide(x, sum, x);
        }

        public static void SiLU(ReadOnlySpan<float> x, Span<float> y)
        {
            for(int i=0; i<x.Length; i++)
            {
                float val = x[i];
                y[i] = val / (1.0f + MathF.Exp(-val));
            }
        }

        public static void Gelu(ReadOnlySpan<float> x, Span<float> y)
        {
             // Tanh approximation
             const float Sqrt2OverPi = 0.7978845608f;
             const float Coeff = 0.044715f;

             for(int i=0; i<x.Length; i++)
             {
                 float val = x[i];
                 float inner = Sqrt2OverPi * (val + Coeff * val * val * val);
                 y[i] = 0.5f * val * (1.0f + MathF.Tanh(inner));
             }
        }

        // y = x @ w.T + b
        // x: [M, K], w: [N, K], b: [N], y: [M, N]
        public static void Linear(ReadOnlySpan<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> b, Span<float> y, int M, int N, int K)
        {
            // Naive parallel implementation
            // Parallelize over M (rows of x)

            // For single-threaded simple implementation first
            for (int i = 0; i < M; i++)
            {
                var rowX = x.Slice(i * K, K);
                var rowY = y.Slice(i * N, N);

                for (int j = 0; j < N; j++)
                {
                    var rowW = w.Slice(j * K, K);
                    float dot = TensorPrimitives.Dot(rowX, rowW);

                    if (!b.IsEmpty)
                    {
                        dot += b[j];
                    }
                    rowY[j] = dot;
                }
            }
        }

        // RoPE
        // x: [seq, heads, head_dim]
        // cos, sin: [seq, head_dim/2]
        // Interleaved style: pairs (0,1), (2,3)
        // x_out[i] = x[i] * cos[i/2] - x[i+1] * sin[i/2]
        // x_out[i+1] = x[i+1] * cos[i/2] + x[i] * sin[i/2]
        public static void ApplyRoPE(Span<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int seqLen, int nHeads, int headDim)
        {
            int halfDim = headDim / 2;

            for (int s = 0; s < seqLen; s++)
            {
                var cosRow = cos.Slice(s * halfDim, halfDim);
                var sinRow = sin.Slice(s * halfDim, halfDim);

                for (int h = 0; h < nHeads; h++)
                {
                    int offset = (s * nHeads + h) * headDim;
                    var head = x.Slice(offset, headDim);

                    for (int i = 0; i < halfDim; i++)
                    {
                        float val0 = head[2 * i];
                        float val1 = head[2 * i + 1];
                        float c = cosRow[i];
                        float sn = sinRow[i];

                        head[2 * i] = val0 * c - val1 * sn;
                        head[2 * i + 1] = val1 * c + val0 * sn;
                    }
                }
            }
        }
    }
}
