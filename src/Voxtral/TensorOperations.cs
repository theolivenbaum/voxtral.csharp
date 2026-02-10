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
            TensorSpan<float> xSpan = x;
            TensorSpan<float> wSpan = w;
            TensorSpan<float> ySpan = y;

            float sumSq = Tensor.Dot<float>(xSpan, xSpan);
            float rms = 1.0f / MathF.Sqrt(sumSq / x.FlattenedLength + eps);

            Tensor.Multiply<float>(xSpan, wSpan, ySpan);
            Tensor.Multiply<float>(ySpan, rms, ySpan);
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
            TensorSpan<float> xSpan = x;

            float max = TensorPrimitives.Max(x.AsSpan());
            Tensor.Subtract<float>(xSpan, max, xSpan);
            Tensor.Exp<float>(xSpan, xSpan);
            float sum = TensorPrimitives.Sum(x.AsSpan());
            Tensor.Multiply<float>(xSpan, 1.0f / sum, xSpan);
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
            TensorSpan<float> xSpan = x;
            TensorSpan<float> ySpan = y;

            Tensor.Sigmoid<float>(xSpan, ySpan);
            Tensor.Multiply<float>(xSpan, ySpan, ySpan);
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
             const float Sqrt2OverPi = 0.7978845608f;
             const float Coeff = 0.044715f;
             const int ChunkSize = 128;

             Span<float> tmpCalcSpan = stackalloc float[ChunkSize];

             // Use flattened spans to avoid multi-dim issues
             Span<float> xSpan = x.AsSpan();
             Span<float> ySpan = y.AsSpan();

             nint len = x.FlattenedLength;
             nint offset = 0;

             while (offset < len)
             {
                 int count = (int)Math.Min(ChunkSize, len - offset);

                 // Create TensorSpans from slices
                 TensorSpan<float> xChunk = new TensorSpan<float>(xSpan.Slice((int)offset, count));
                 TensorSpan<float> yChunk = new TensorSpan<float>(ySpan.Slice((int)offset, count));
                 TensorSpan<float> tCalc = new TensorSpan<float>(tmpCalcSpan.Slice(0, count));

                 // tCalc = x^3
                 xChunk.CopyTo(tCalc);
                 Tensor.Multiply<float>(tCalc, tCalc, tCalc); // x^2
                 Tensor.Multiply<float>(tCalc, xChunk, tCalc); // x^3

                 // tCalc = tCalc * Coeff + x
                 Tensor.Multiply<float>(tCalc, Coeff, tCalc);
                 Tensor.Add<float>(tCalc, xChunk, tCalc);

                 // tCalc = Tanh(tCalc * Sqrt2OverPi)
                 Tensor.Multiply<float>(tCalc, Sqrt2OverPi, tCalc);
                 Tensor.Tanh<float>(tCalc, tCalc);

                 // y = 0.5 * x * (1 + tCalc)
                 Tensor.Add<float>(tCalc, 1.0f, tCalc);
                 Tensor.Multiply<float>(tCalc, xChunk, tCalc);
                 Tensor.Multiply<float>(tCalc, 0.5f, yChunk);

                 offset += count;
             }
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

                    int bLen = b.Length;

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

                            for (int j = 0; j < N; j++)
                            {
                                ReadOnlySpan<float> rowW = new ReadOnlySpan<float>(pWLocal + j * K, K);

                                float dot = TensorPrimitives.Dot(rowX, rowW);

                                if (pBLocal != null && bLen > 0)
                                {
                                    dot += pBLocal[j];
                                }
                                (pYLocal + i * N)[j] = dot;
                            }
                        });
                    }
                }
            }
        }

        public static void Linear(Tensor<float> x, Tensor<float> w, Tensor<float>? b, Tensor<float> y)
        {
            TensorSpan<float> ws = w;
            nint K = ws.Lengths[1];
            nint N = ws.Lengths[0];
            nint xLen = x.FlattenedLength;
            nint M = xLen / K;

            Parallel.For(0, (int)((M < 4) ? N : M), idx =>
            {
                // Use flattened spans
                Span<float> xSpan = x.AsSpan();
                Span<float> wSpan = w.AsSpan();
                Span<float> bSpan = (b is not null) ? b.AsSpan() : default;
                Span<float> ySpan = y.AsSpan();

                if (M < 4)
                {
                    nint j = idx;
                    var rowW = wSpan.Slice((int)(j * K), (int)K);
                    TensorSpan<float> rowWTs = new TensorSpan<float>(rowW);

                    float bias = (!bSpan.IsEmpty) ? bSpan[(int)j] : 0.0f;

                    for (nint i = 0; i < M; i++)
                    {
                        var rowX = xSpan.Slice((int)(i * K), (int)K);
                        TensorSpan<float> rowXTs = new TensorSpan<float>(rowX);
                        float dot = Tensor.Dot<float>(rowXTs, rowWTs);
                        ySpan[(int)(i * N + j)] = dot + bias;
                    }
                }
                else
                {
                    nint i = idx;
                    var rowX = xSpan.Slice((int)(i * K), (int)K);
                    TensorSpan<float> rowXTs = new TensorSpan<float>(rowX);

                    for (nint j = 0; j < N; j++)
                    {
                        var rowW = wSpan.Slice((int)(j * K), (int)K);
                        TensorSpan<float> rowWTs = new TensorSpan<float>(rowW);
                        float dot = Tensor.Dot<float>(rowXTs, rowWTs);
                        if (!bSpan.IsEmpty)
                        {
                            dot += bSpan[(int)j];
                        }
                        ySpan[(int)(i * N + j)] = dot;
                    }
                }
            });
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
            int halfDim = headDim / 2;

            Parallel.For(0, (seqLen < 4) ? nHeads : seqLen, idx =>
            {
                Span<float> xSpan = x.AsSpan();
                Span<float> cosSpan = cos.AsSpan();
                Span<float> sinSpan = sin.AsSpan();

                if (seqLen < 4)
                {
                    int h = idx;
                    for (int s = 0; s < seqLen; s++)
                    {
                        var cosRow = cosSpan.Slice(s * halfDim, halfDim);
                        var sinRow = sinSpan.Slice(s * halfDim, halfDim);

                        nint offset = (nint)((long)(s * nHeads + h) * headDim);

                        for (int i = 0; i < halfDim; i++)
                        {
                            nint idx0 = offset + 2 * i;
                            nint idx1 = offset + 2 * i + 1;

                            float val0 = xSpan[(int)idx0];
                            float val1 = xSpan[(int)idx1];
                            float c = cosRow[i];
                            float sn = sinRow[i];

                            xSpan[(int)idx0] = val0 * c - val1 * sn;
                            xSpan[(int)idx1] = val1 * c + val0 * sn;
                        }
                    }
                }
                else
                {
                    int s = idx;
                    var cosRow = cosSpan.Slice(s * halfDim, halfDim);
                    var sinRow = sinSpan.Slice(s * halfDim, halfDim);

                    for (int h = 0; h < nHeads; h++)
                    {
                        nint offset = (nint)((long)(s * nHeads + h) * headDim);

                        for (int i = 0; i < halfDim; i++)
                        {
                            nint idx0 = offset + 2 * i;
                            nint idx1 = offset + 2 * i + 1;

                            float val0 = xSpan[(int)idx0];
                            float val1 = xSpan[(int)idx1];
                            float c = cosRow[i];
                            float sn = sinRow[i];

                            xSpan[(int)idx0] = val0 * c - val1 * sn;
                            xSpan[(int)idx1] = val1 * c + val0 * sn;
                        }
                    }
                }
            });
        }
    }
}
