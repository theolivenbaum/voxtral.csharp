using System;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx.DirectML
{
    public class OnnxAdapter
    {
        private readonly DenseTensor<float> _w0;
        private readonly DenseTensor<float> _w1;

        private const int DOWNSAMPLE = 4;
        private const int ENC_DIM = 1280;
        private const int ADAPTER_HIDDEN = 5120;
        private const int DEC_DIM = 3072;
        private const int HIDDEN_DIM = 3072;

        public OnnxAdapter(OnnxSafetensorsReader reader)
        {
            string p = "mm_streams_embeddings.embedding_module.audio_language_projection";
            // Check paths. Original code: "mm_streams_embeddings.embedding_module.audio_language_projection"
            // Then ".0.weight" and ".2.weight".
            _w0 = reader.LoadTensor($"{p}.0.weight");
            _w1 = reader.LoadTensor($"{p}.2.weight");
        }

        public DenseTensor<float> Forward(DenseTensor<float> encOut)
        {
            // encOut: [seq * 1280] or [seq, 1280]
            long flattenedLength = encOut.Buffer.Length;
            int seqLen = (int)(flattenedLength / ENC_DIM);
            if (seqLen % DOWNSAMPLE != 0)
                throw new Exception($"Encoder output length {seqLen} not divisible by {DOWNSAMPLE}");

            int outLen = seqLen / DOWNSAMPLE;

            // Reshape [seq, 1280] -> [outLen, 5120] handled implicitly by Linear treating input as flat buffer

            // Linear 0: [outLen, 5120] -> [outLen, 3072]
            // hidden shape: [outLen, 3072]
            float[] hiddenData = new float[outLen * HIDDEN_DIM];
            DenseTensor<float> hidden = new DenseTensor<float>(hiddenData, new int[] { outLen, HIDDEN_DIM });

            // _w0 shape: [3072, 5120] (N=3072, K=5120)
            OnnxTensorOperations.Linear(encOut, _w0, null, hidden);

            // GELU
            OnnxTensorOperations.Gelu(hidden, hidden);

            // Linear 1: [outLen, 3072] -> [outLen, 3072]
            // output shape: [outLen, 3072] (DEC_DIM=3072)
            float[] outData = new float[outLen * DEC_DIM];
            DenseTensor<float> output = new DenseTensor<float>(outData, new int[] { outLen, DEC_DIM });

            OnnxTensorOperations.Linear(hidden, _w1, null, output);

            return output;
        }
    }
}
