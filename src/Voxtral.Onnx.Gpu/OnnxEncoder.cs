using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx.Gpu
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

            var dims = mel.Dimensions;
            var batchedMel = new DenseTensor<float>(mel.Buffer, new int[] { 1, dims[0], dims[1] });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, batchedMel)
            };

            using var results = _session.Run(inputs);
            var output = results.First();
            var outputTensor = output.AsTensor<float>().ToDenseTensor();

            if (outputTensor.Dimensions.Length == 3 && outputTensor.Dimensions[0] == 1)
            {
                var d = outputTensor.Dimensions;
                seqLen = d[1];
                return new DenseTensor<float>(outputTensor.Buffer, new int[] { d[1], d[2] });
            }

            seqLen = outputTensor.Dimensions[0];
            return outputTensor;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
