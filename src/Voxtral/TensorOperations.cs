using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Voxtral
{
    public static class TensorOperations
    {
        public static void RMSNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> y, float eps)
        {
            // x: [dim], w: [dim], y: [dim]
            float sumSq = TensorPrimitives.SumOfSquares(x);
            float rms = 1.0f / MathF.Sqrt(sumSq / x.Length + eps);

            if (x.Length == w.Length && x.Length == y.Length)
            {
                int i = 0;
                int len = x.Length;
                int vectorSize = Vector<float>.Count;
                int vectorLimit = len - (len % vectorSize);

                for (; i < vectorLimit; i += vectorSize)
                {
                    Vector<float> vx = new Vector<float>(x.Slice(i));
                    Vector<float> vw = new Vector<float>(w.Slice(i));
                    Vector<float> vy = vx * vw * rms;
                    vy.CopyTo(y.Slice(i));
                }

                for (; i < len; i++)
                {
                    y[i] = x[i] * w[i] * rms;
                }
            }
            else
            {
                TensorPrimitives.Multiply(x, w, y);
                TensorPrimitives.Multiply(y, rms, y);
            }
        }

        public static void RMSNorm(Tensor<float> x, Tensor<float> w, Tensor<float> y, float eps)
        {
            RMSNorm(x.AsSpan(), w.AsSpan(), y.AsSpan(), eps);
        }

        public static void Softmax(Span<float> x)
        {
            float max = TensorPrimitives.Max(x);
            TensorPrimitives.Subtract(x, max, x);
            TensorPrimitives.Exp(x, x);
            float sum = TensorPrimitives.Sum(x);
            TensorPrimitives.Divide(x, sum, x);
        }

        public static void Softmax(Tensor<float> x)
        {
            Softmax(x.AsSpan());
        }

        public static void SiLU(ReadOnlySpan<float> x, Span<float> y)
        {
            const int ChunkSize = 128;
            Span<float> tmpX = stackalloc float[ChunkSize];
            Span<float> tmpCalc = stackalloc float[ChunkSize];

            int offset = 0;
            while (offset < x.Length)
            {
                int count = Math.Min(ChunkSize, x.Length - offset);
                var xChunk = x.Slice(offset, count);
                var yChunk = y.Slice(offset, count);
                var tX = tmpX.Slice(0, count);
                var tCalc = tmpCalc.Slice(0, count);

                xChunk.CopyTo(tX);
                TensorPrimitives.Sigmoid(tX, tCalc);
                TensorPrimitives.Multiply(tX, tCalc, yChunk);

                offset += count;
            }
        }

        public static void SiLU(Tensor<float> x, Tensor<float> y)
        {
            SiLU(x.AsSpan(), y.AsSpan());
        }

        public static void Gelu(ReadOnlySpan<float> x, Span<float> y)
        {
             const float Sqrt2OverPi = 0.7978845608f;
             const float Coeff = 0.044715f;
             const int ChunkSize = 128;

             Span<float> tmpX = stackalloc float[ChunkSize];
             Span<float> tmpCalc = stackalloc float[ChunkSize];

             int offset = 0;
             while (offset < x.Length)
             {
                 int count = Math.Min(ChunkSize, x.Length - offset);
                 var xChunk = x.Slice(offset, count);
                 var yChunk = y.Slice(offset, count);
                 var tX = tmpX.Slice(0, count);
                 var tCalc = tmpCalc.Slice(0, count);

                 xChunk.CopyTo(tX);

                 tX.CopyTo(tCalc);
                 TensorPrimitives.Multiply(tCalc, tX, tCalc); // x^2
                 TensorPrimitives.Multiply(tCalc, tX, tCalc); // x^3

                 TensorPrimitives.Multiply(tCalc, Coeff, tCalc);
                 TensorPrimitives.Add(tCalc, tX, tCalc);

                 TensorPrimitives.Multiply(tCalc, Sqrt2OverPi, tCalc);
                 TensorPrimitives.Tanh(tCalc, tCalc);

                 TensorPrimitives.Add(tCalc, 1.0f, tCalc);
                 TensorPrimitives.Multiply(tCalc, tX, tCalc);
                 TensorPrimitives.Multiply(tCalc, 0.5f, yChunk);

                 offset += count;
             }
        }

        public static void Gelu(Tensor<float> x, Tensor<float> y)
        {
            Gelu(x.AsSpan(), y.AsSpan());
        }

        public static void Linear(ReadOnlySpan<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> b, Span<float> y, int M, int N, int K)
        {
            bool hasBias = !b.IsEmpty;

            // Standard sequential implementation (no unsafe, no Parallel.For)
            for (int i = 0; i < M; i++)
            {
                ReadOnlySpan<float> rowX = x.Slice(i * K, K);
                Span<float> rowY = y.Slice(i * N, N);

                for (int j = 0; j < N; j++)
                {
                    ReadOnlySpan<float> rowW = w.Slice(j * K, K);

                    float dot = TensorPrimitives.Dot(rowX, rowW);

                    if (hasBias)
                    {
                        dot += b[j];
                    }

                    rowY[j] = dot;
                }
            }
        }

        public static void Linear(Tensor<float> x, Tensor<float> w, Tensor<float>? b, Tensor<float> y)
        {
            // x: [M, K]
            // w: [N, K]
            // y: [M, N]
            TensorSpan<float> xSpan = x;
            TensorSpan<float> wSpan = w;
            TensorSpan<float> ySpan = y;

            // Check dimensions
            int M = (int)x.Lengths[0];
            int K = (int)x.Lengths[1];
            int N = (int)w.Lengths[0];

            // Bias handling
            ReadOnlySpan<float> bSpan = b != null ? b.AsSpan() : ReadOnlySpan<float>.Empty;
            bool hasBias = !bSpan.IsEmpty;

            // Stack allocate index arrays to avoid heap allocations
            Span<nint> xIndices = stackalloc nint[2];
            Span<nint> wIndices = stackalloc nint[2];
            Span<nint> yIndices = stackalloc nint[2];

            // Iterate sequentially (no Parallel.For as requested)
            for (int i = 0; i < M; i++)
            {
                xIndices[0] = i;
                xIndices[1] = 0;
                var rowX = xSpan.GetSpan(xIndices, K);

                // Prepare Y row access
                yIndices[0] = i;
                yIndices[1] = 0;
                var rowY = ySpan.GetSpan(yIndices, N);

                for (int j = 0; j < N; j++)
                {
                    wIndices[0] = j;
                    wIndices[1] = 0;
                    var rowW = wSpan.GetSpan(wIndices, K);

                    float dot = TensorPrimitives.Dot(rowX, rowW);

                    if (hasBias)
                    {
                        dot += bSpan[j];
                    }

                    rowY[j] = dot;
                }
            }
        }

        public static void ApplyRoPE(Span<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int seqLen, int nHeads, int headDim)
        {
            int halfDim = headDim / 2;

            unsafe
            {
                fixed (float* px = x)
                fixed (float* pCos = cos)
                fixed (float* pSin = sin)
                {
                    nint ptrX = (nint)px;
                    nint ptrCos = (nint)pCos;
                    nint ptrSin = (nint)pSin;

                    if (seqLen < 4)
                    {
                        Parallel.For(0, nHeads, h =>
                        {
                            float* pXLocal = (float*)ptrX;
                            float* pCosLocal = (float*)ptrCos;
                            float* pSinLocal = (float*)ptrSin;

                            for (int s = 0; s < seqLen; s++)
                            {
                                ReadOnlySpan<float> cosRow = new ReadOnlySpan<float>(pCosLocal + s * halfDim, halfDim);
                                ReadOnlySpan<float> sinRow = new ReadOnlySpan<float>(pSinLocal + s * halfDim, halfDim);

                                int offset = (s * nHeads + h) * headDim;
                                float* pHead = pXLocal + offset;

                                for (int i = 0; i < halfDim; i++)
                                {
                                    float val0 = pHead[2 * i];
                                    float val1 = pHead[2 * i + 1];
                                    float c = cosRow[i];
                                    float sn = sinRow[i];

                                    pHead[2 * i] = val0 * c - val1 * sn;
                                    pHead[2 * i + 1] = val1 * c + val0 * sn;
                                }
                            }
                        });
                    }
                    else
                    {
                        Parallel.For(0, seqLen, s =>
                        {
                            float* pXLocal = (float*)ptrX;
                            float* pCosLocal = (float*)ptrCos;
                            float* pSinLocal = (float*)ptrSin;

                            ReadOnlySpan<float> cosRow = new ReadOnlySpan<float>(pCosLocal + s * halfDim, halfDim);
                            ReadOnlySpan<float> sinRow = new ReadOnlySpan<float>(pSinLocal + s * halfDim, halfDim);

                            for (int h = 0; h < nHeads; h++)
                            {
                                int offset = (s * nHeads + h) * headDim;
                                float* pHead = pXLocal + offset;

                                for (int i = 0; i < halfDim; i++)
                                {
                                    float val0 = pHead[2 * i];
                                    float val1 = pHead[2 * i + 1];
                                    float c = cosRow[i];
                                    float sn = sinRow[i];

                                    pHead[2 * i] = val0 * c - val1 * sn;
                                    pHead[2 * i + 1] = val1 * c + val0 * sn;
                                }
                            }
                        });
                    }
                }
            }
        }

        public static void ApplyRoPE(Tensor<float> x, Tensor<float> cos, Tensor<float> sin, int seqLen, int nHeads, int headDim)
        {
            ApplyRoPE(x.AsSpan(), cos.AsSpan(), sin.AsSpan(), seqLen, nHeads, headDim);
        }
    }
}
