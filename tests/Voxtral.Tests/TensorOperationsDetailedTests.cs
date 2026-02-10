using System;
using System.Numerics;
using System.Numerics.Tensors;
using Xunit;

namespace Voxtral.Tests
{
    public class TensorOperationsDetailedTests
    {
        [Fact]
        public void RMSNorm_Test()
        {
            int dim = 100;
            float[] x = new float[dim];
            float[] w = new float[dim];
            float[] y = new float[dim];
            float eps = 1e-5f;

            var rnd = new Random(42);
            for(int i=0; i<dim; i++) { x[i] = (float)rnd.NextDouble(); w[i] = (float)rnd.NextDouble(); }

            // Reference implementation
            float sumSq = 0;
            for(int i=0; i<dim; i++) sumSq += x[i]*x[i];
            float rms = 1.0f / MathF.Sqrt(sumSq / dim + eps);
            float[] expected = new float[dim];
            for(int i=0; i<dim; i++) expected[i] = x[i] * w[i] * rms;

            TensorOperations.RMSNorm(x, w, y, eps);

            for(int i=0; i<dim; i++) Assert.Equal(expected[i], y[i], 1e-5f);
        }

        [Fact]
        public void Softmax_Test()
        {
            int dim = 50;
            float[] x = new float[dim];
            var rnd = new Random(43);
            for(int i=0; i<dim; i++) x[i] = (float)rnd.NextDouble();

            float[] expected = new float[dim];
            Array.Copy(x, expected, dim);

            // Reference
            float max = expected[0];
            for(int i=1; i<dim; i++) if(expected[i] > max) max = expected[i];
            float sum = 0;
            for(int i=0; i<dim; i++) {
                expected[i] = MathF.Exp(expected[i] - max);
                sum += expected[i];
            }
            for(int i=0; i<dim; i++) expected[i] /= sum;

            TensorOperations.Softmax(x);

            for(int i=0; i<dim; i++) Assert.Equal(expected[i], x[i], 1e-5f);
        }

        [Fact]
        public void SiLU_Test()
        {
            int dim = 100;
            float[] x = new float[dim];
            float[] y = new float[dim];
            var rnd = new Random(44);
            for(int i=0; i<dim; i++) x[i] = (float)rnd.NextDouble() * 10 - 5;

            float[] expected = new float[dim];
            for(int i=0; i<dim; i++) {
                expected[i] = x[i] / (1.0f + MathF.Exp(-x[i]));
            }

            TensorOperations.SiLU(x, y);

            for(int i=0; i<dim; i++) Assert.Equal(expected[i], y[i], 1e-5f);
        }

        [Fact]
        public void Gelu_Test()
        {
            int dim = 100;
            float[] x = new float[dim];
            float[] y = new float[dim];
            var rnd = new Random(45);
            for(int i=0; i<dim; i++) x[i] = (float)rnd.NextDouble() * 4 - 2;

            float[] expected = new float[dim];
            const float Sqrt2OverPi = 0.7978845608f;
            const float Coeff = 0.044715f;
            for(int i=0; i<dim; i++) {
                 float val = x[i];
                 float inner = Sqrt2OverPi * (val + Coeff * val * val * val);
                 expected[i] = 0.5f * val * (1.0f + MathF.Tanh(inner));
            }

            TensorOperations.Gelu(x, y);

            for(int i=0; i<dim; i++) Assert.Equal(expected[i], y[i], 1e-5f);
        }

        [Fact]
        public void Linear_Test()
        {
            int M = 10, N = 20, K = 30;
            float[] x = new float[M * K];
            float[] w = new float[N * K];
            float[] b = new float[N];
            float[] y = new float[M * N];

            var rnd = new Random(46);
            for(int i=0; i<x.Length; i++) x[i] = (float)rnd.NextDouble();
            for(int i=0; i<w.Length; i++) w[i] = (float)rnd.NextDouble();
            for(int i=0; i<b.Length; i++) b[i] = (float)rnd.NextDouble();

            // Reference
            float[] expected = new float[M * N];
            for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                    float dot = 0;
                    for(int k=0; k<K; k++) {
                        dot += x[i*K + k] * w[j*K + k];
                    }
                    expected[i*N + j] = dot + b[j];
                }
            }

            TensorOperations.Linear(x, w, b, y, M, N, K);

            for(int i=0; i<expected.Length; i++) Assert.Equal(expected[i], y[i], 1e-4f);
        }

        [Fact]
        public void ApplyRoPE_Test()
        {
            int seqLen = 5;
            int nHeads = 2;
            int headDim = 4; // halfDim = 2
            int halfDim = 2;
            int dim = seqLen * nHeads * headDim;

            float[] x = new float[dim];
            float[] cos = new float[seqLen * halfDim];
            float[] sin = new float[seqLen * halfDim];

            var rnd = new Random(47);
            for(int i=0; i<dim; i++) x[i] = (float)rnd.NextDouble();
            for(int i=0; i<cos.Length; i++) cos[i] = (float)rnd.NextDouble();
            for(int i=0; i<sin.Length; i++) sin[i] = (float)rnd.NextDouble();

            float[] expected = (float[])x.Clone();

            // Reference
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < nHeads; h++)
                {
                    int offset = (s * nHeads + h) * headDim;
                    for (int i = 0; i < halfDim; i++)
                    {
                        float val0 = expected[offset + 2 * i];
                        float val1 = expected[offset + 2 * i + 1];
                        float c = cos[s * halfDim + i];
                        float sn = sin[s * halfDim + i];

                        expected[offset + 2 * i] = val0 * c - val1 * sn;
                        expected[offset + 2 * i + 1] = val1 * c + val0 * sn;
                    }
                }
            }

            TensorOperations.ApplyRoPE(x, cos, sin, seqLen, nHeads, headDim);

            for(int i=0; i<dim; i++) Assert.Equal(expected[i], x[i], 1e-5f);
        }
    }
}
