using System;
using System.Numerics.Tensors;

namespace Voxtral
{
    public class Adapter
    {
        private readonly float[] _w0, _w1;

        private const int DOWNSAMPLE = 4;
        private const int ENC_DIM = 1280;
        private const int ADAPTER_HIDDEN = 5120;
        private const int DEC_DIM = 3072;
        private const int HIDDEN_DIM = 3072;

        public Adapter(SafetensorsReader reader)
        {
            string p = "mm_streams_embeddings.embedding_module.audio_language_projection";
            _w0 = reader.LoadTensor($"{p}.0.weight").ToArray();
            _w1 = reader.LoadTensor($"{p}.2.weight").ToArray();
        }

        public float[] Forward(float[] encOut)
        {
            // encOut: [seq * 1280]
            int seqLen = encOut.Length / ENC_DIM;
            if (seqLen % DOWNSAMPLE != 0)
                throw new Exception($"Encoder output length {seqLen} not divisible by {DOWNSAMPLE}");

            int outLen = seqLen / DOWNSAMPLE;

            // Reshape [seq, 1280] -> [outLen, 5120]
            var encSpan = encOut.AsSpan();

            float[] hidden = new float[outLen * HIDDEN_DIM];

            // Linear 0: [outLen, 5120] -> [outLen, 3072]
            TensorOperations.Linear(encSpan, _w0, ReadOnlySpan<float>.Empty, hidden, outLen, HIDDEN_DIM, ADAPTER_HIDDEN);

            // GELU
            TensorOperations.Gelu(hidden, hidden);

            // Linear 1: [outLen, 3072] -> [outLen, 3072]
            float[] output = new float[outLen * DEC_DIM];
            TensorOperations.Linear(hidden, _w1, ReadOnlySpan<float>.Empty, output, outLen, DEC_DIM, HIDDEN_DIM);

            return output;
        }
    }
}
