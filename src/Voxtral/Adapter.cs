using System;
using System.Numerics.Tensors;

namespace Voxtral
{
    public class Adapter
    {
        private readonly Tensor<float> _w0, _w1;

        private const int DOWNSAMPLE = 4;
        private const int ENC_DIM = 1280;
        private const int ADAPTER_HIDDEN = 5120;
        private const int DEC_DIM = 3072;
        private const int HIDDEN_DIM = 3072;

        public Adapter(SafetensorsReader reader)
        {
            string p = "mm_streams_embeddings.embedding_module.audio_language_projection";
            _w0 = reader.LoadTensor($"{p}.0.weight");
            _w1 = reader.LoadTensor($"{p}.2.weight");
        }

        public Tensor<float> Forward(Tensor<float> encOut)
        {
            // encOut: [seq * 1280] or [seq, 1280]
            TensorSpan<float> encSpan = encOut;
            int seqLen = (int)(encSpan.FlattenedLength / ENC_DIM);
            if (seqLen % DOWNSAMPLE != 0)
                throw new Exception($"Encoder output length {seqLen} not divisible by {DOWNSAMPLE}");

            int outLen = seqLen / DOWNSAMPLE;

            // Reshape [seq, 1280] -> [outLen, 5120] handled implicitly by Linear treating input as flat buffer

            // Linear 0: [outLen, 5120] -> [outLen, 3072]
            // hidden shape: [outLen, 3072]
            float[] hiddenData = new float[outLen * HIDDEN_DIM];
            Tensor<float> hidden = Tensor.Create(hiddenData, new nint[] { outLen, HIDDEN_DIM });

            // _w0 shape: [3072, 5120] (N=3072, K=5120)
            TensorOperations.Linear(encOut, _w0, null, hidden);

            // GELU
            TensorOperations.Gelu(hidden, hidden);

            // Linear 1: [outLen, 3072] -> [outLen, 3072]
            // output shape: [outLen, 3072] (DEC_DIM=3072)
            float[] outData = new float[outLen * DEC_DIM];
            Tensor<float> output = Tensor.Create(outData, new nint[] { outLen, DEC_DIM });

            TensorOperations.Linear(hidden, _w1, null, output);

            return output;
        }
    }
}
