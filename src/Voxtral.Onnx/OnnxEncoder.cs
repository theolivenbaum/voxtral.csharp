using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx
{
    public class OnnxEncoder : IDisposable
    {
        private readonly InferenceSession _session;

        public OnnxEncoder(string modelPath, SessionOptions? options = null)
        {
            _session = new InferenceSession(modelPath, options ?? new SessionOptions());
        }

        public DenseTensor<float> Forward(DenseTensor<float> mel, out int seqLen)
        {
            string inputName = "mel";
            if (_session.InputMetadata.Count > 0)
            {
                inputName = _session.InputMetadata.Keys.First();
            }

            // Reshape [128, frames] to [1, 128, frames]
            var dims = mel.Dimensions;
            var batchedMel = new DenseTensor<float>(mel.Buffer, new int[] { 1, dims[0], dims[1] });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, batchedMel)
            };

            using var results = _session.Run(inputs);
            var output = results.First();
            var outputTensor = output.AsTensor<float>().ToDenseTensor();

            // Squeeze batch dimension [1, seqLen, dim] -> [seqLen, dim]
            if (outputTensor.Dimensions.Length == 3 && outputTensor.Dimensions[0] == 1)
            {
                var d = outputTensor.Dimensions;
                seqLen = d[1];
                return new DenseTensor<float>(outputTensor.Buffer, new int[] { d[1], d[2] });
            }

            seqLen = outputTensor.Dimensions[0]; // Assumption
            return outputTensor;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
