using System;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using Voxtral;

namespace Voxtral.Tests
{
    public class PerformanceTests
    {
        private readonly ITestOutputHelper _output;

        public PerformanceTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void BenchmarkLinear_Decoding()
        {
            // M=1, N=4096, K=3072
            int M = 1;
            int N = 4096;
            int K = 3072;

            float[] x = new float[M * K];
            float[] w = new float[N * K];
            float[] y = new float[M * N];
            float[] b = new float[0];

            // Warmup
            TensorOperations.Linear(x, w, b, y, M, N, K);

            Stopwatch sw = Stopwatch.StartNew();
            int iterations = 100;
            for (int i = 0; i < iterations; i++)
            {
                TensorOperations.Linear(x, w, b, y, M, N, K);
            }
            sw.Stop();

            _output.WriteLine($"Linear (M={M}, N={N}, K={K}) - {iterations} iterations: {sw.Elapsed.TotalMilliseconds} ms");
            _output.WriteLine($"Avg time: {sw.Elapsed.TotalMilliseconds / iterations} ms");
        }

        [Fact]
        public void BenchmarkLinear_Prefill()
        {
            // M=128, N=4096, K=3072
            int M = 128;
            int N = 4096;
            int K = 3072;

            float[] x = new float[M * K];
            float[] w = new float[N * K];
            float[] y = new float[M * N];
            float[] b = new float[0];

            // Warmup
            TensorOperations.Linear(x, w, b, y, M, N, K);

            Stopwatch sw = Stopwatch.StartNew();
            int iterations = 10;
            for (int i = 0; i < iterations; i++)
            {
                TensorOperations.Linear(x, w, b, y, M, N, K);
            }
            sw.Stop();

            _output.WriteLine($"Linear (M={M}, N={N}, K={K}) - {iterations} iterations: {sw.Elapsed.TotalMilliseconds} ms");
            _output.WriteLine($"Avg time: {sw.Elapsed.TotalMilliseconds / iterations} ms");
        }

        [Fact]
        public void BenchmarkApplyRoPE()
        {
            int seqLen = 1;
            int nHeads = 32;
            int headDim = 128;
            int dim = nHeads * headDim;

            float[] x = new float[seqLen * dim];
            float[] cos = new float[seqLen * headDim / 2];
            float[] sin = new float[seqLen * headDim / 2];

            // Warmup
            TensorOperations.ApplyRoPE(x, cos, sin, seqLen, nHeads, headDim);

            Stopwatch sw = Stopwatch.StartNew();
            int iterations = 10000;
            for (int i = 0; i < iterations; i++)
            {
                TensorOperations.ApplyRoPE(x, cos, sin, seqLen, nHeads, headDim);
            }
            sw.Stop();

             _output.WriteLine($"ApplyRoPE (seqLen={seqLen}, nHeads={nHeads}) - {iterations} iterations: {sw.Elapsed.TotalMilliseconds} ms");
             _output.WriteLine($"Avg time: {sw.Elapsed.TotalMilliseconds / iterations} ms");
        }
    }
}
