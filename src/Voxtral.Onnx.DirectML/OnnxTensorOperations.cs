using System;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx.DirectML
{
    public static class OnnxTensorOperations
    {
        public static void RMSNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> y, float eps)
        {
            // Implementation is identical to TensorOperations.RMSNorm
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

        public static void RMSNorm(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float> y, float eps)
        {
            RMSNorm(x.Buffer.Span, w.Buffer.Span, y.Buffer.Span, eps);
        }

        public static void Softmax(Span<float> x)
        {
            float max = TensorPrimitives.Max(x);
            TensorPrimitives.Subtract(x, max, x);
            TensorPrimitives.Exp(x, x);
            float sum = TensorPrimitives.Sum(x);
            TensorPrimitives.Divide(x, sum, x);
        }

        public static void Softmax(DenseTensor<float> x)
        {
            Softmax(x.Buffer.Span);
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

        public static void SiLU(DenseTensor<float> x, DenseTensor<float> y)
        {
            SiLU(x.Buffer.Span, y.Buffer.Span);
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

        public static void Gelu(DenseTensor<float> x, DenseTensor<float> y)
        {
            Gelu(x.Buffer.Span, y.Buffer.Span);
        }

        public static void Linear(ReadOnlySpan<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> b, Span<float> y, int M, int N, int K)
        {
            unsafe
            {
                fixed (float* px = x)
                fixed (float* pw = w)
                fixed (float* pb = b)
                fixed (float* py = y)
                {
                    nint ptrX = (nint)px;
                    nint ptrW = (nint)pw;
                    nint ptrB = (nint)pb;
                    nint ptrY = (nint)py;

                    int bLen = b.Length; // Capture length

                    if (M < 4)
                    {
                        Parallel.For(0, N, j =>
                        {
                            float* pXLocal = (float*)ptrX;
                            float* pWLocal = (float*)ptrW;
                            float* pBLocal = (float*)ptrB;
                            float* pYLocal = (float*)ptrY;

                            ReadOnlySpan<float> rowW = new ReadOnlySpan<float>(pWLocal + j * K, K);
                            float bias = (pBLocal != null && bLen > 0) ? pBLocal[j] : 0.0f;

                            for (int i = 0; i < M; i++)
                            {
                                ReadOnlySpan<float> rowX = new ReadOnlySpan<float>(pXLocal + i * K, K);
                                float dot = TensorPrimitives.Dot(rowX, rowW);
                                (pYLocal + i * N)[j] = dot + bias;
                            }
                        });
                    }
                    else
                    {
                        Parallel.For(0, M, i =>
                        {
                            float* pXLocal = (float*)ptrX;
                            float* pWLocal = (float*)ptrW;
                            float* pBLocal = (float*)ptrB;
                            float* pYLocal = (float*)ptrY;

                            ReadOnlySpan<float> rowX = new ReadOnlySpan<float>(pXLocal + i * K, K);
                            // RowY starts at pYLocal + i * N

                            for (int j = 0; j < N; j++)
                            {
                                ReadOnlySpan<float> rowW = new ReadOnlySpan<float>(pWLocal + j * K, K);

                                float dot = TensorPrimitives.Dot(rowX, rowW);

                                if (pBLocal != null && bLen > 0)
                                {
                                    dot += pBLocal[j];
                                }
                                // Store result
                                (pYLocal + i * N)[j] = dot;
                            }
                        });
                    }
                }
            }
        }

        public static void Linear(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float>? b, DenseTensor<float> y)
        {
            // w is [N, K]
            var dims = w.Dimensions;
            int K = dims[1];
            int N = dims[0];
            int M = (int)(x.Buffer.Length / K);

            Linear(x.Buffer.Span, w.Buffer.Span, b != null ? b.Buffer.Span : ReadOnlySpan<float>.Empty, y.Buffer.Span, M, N, K);
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

        public static void ApplyRoPE(DenseTensor<float> x, DenseTensor<float> cos, DenseTensor<float> sin, int seqLen, int nHeads, int headDim)
        {
            ApplyRoPE(x.Buffer.Span, cos.Buffer.Span, sin.Buffer.Span, seqLen, nHeads, headDim);
        }
    }
}
