using System;
using Voxtral;
using Xunit;

namespace Voxtral.Tests
{
    public class TensorOperationsTests
    {
        [Fact]
        public void RMSNorm_ComputesCorrectly()
        {
            // x = [1, 2, 3]
            // w = [0.5, 0.5, 0.5]
            // mean(x^2) = (1+4+9)/3 = 14/3 = 4.6666...
            // rms = 1 / sqrt(4.6666... + eps)
            // expected y = x * rms * w

            float[] x = { 1.0f, 2.0f, 3.0f };
            float[] w = { 0.5f, 0.5f, 0.5f };
            float[] y = new float[3];
            float eps = 1e-5f;

            float meanSq = (1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f) / 3.0f;
            float rms = 1.0f / MathF.Sqrt(meanSq + eps);

            float[] expected = new float[3];
            for (int i = 0; i < 3; i++)
            {
                expected[i] = x[i] * rms * w[i];
            }

            TensorOperations.RMSNorm(x, w, y, eps);

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(expected[i], y[i], 0.0001f); // Check float equality
            }
        }

        [Fact]
        public void RMSNorm_WithNonUniformWeights_ComputesCorrectly()
        {
            // x = [1, 2]
            // w = [0.5, 2.0]
            // mean(x^2) = (1+4)/2 = 2.5
            // rms = 1 / sqrt(2.5 + eps)
            // expected y = [1*rms*0.5, 2*rms*2.0]

            float[] x = { 1.0f, 2.0f };
            float[] w = { 0.5f, 2.0f };
            float[] y = new float[2];
            float eps = 1e-5f;

            float meanSq = 2.5f;
            float rms = 1.0f / MathF.Sqrt(meanSq + eps);

            float[] expected = { 1.0f * rms * 0.5f, 2.0f * rms * 2.0f };

            TensorOperations.RMSNorm(x, w, y, eps);

            Assert.Equal(expected[0], y[0], 0.0001f);
            Assert.Equal(expected[1], y[1], 0.0001f);
        }
    }
}
