using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx.Gpu
{
    public class OnnxAdapter : IDisposable
    {
        private readonly InferenceSession _session;

        public OnnxAdapter(string modelPath, SessionOptions? options = null)
        {
            _session = new InferenceSession(modelPath, options ?? new SessionOptions());
        }

        public DenseTensor<float> Forward(DenseTensor<float> inputs)
        {
            string inputName = "hidden_states";
            if (_session.InputMetadata.Count > 0)
            {
                inputName = _session.InputMetadata.Keys.First();
            }

            var dims = inputs.Dimensions;
            DenseTensor<float> batchedInput;
            if (dims.Length == 2)
            {
                batchedInput = new DenseTensor<float>(inputs.Buffer, new int[] { 1, dims[0], dims[1] });
            }
            else
            {
                batchedInput = inputs;
            }

            var inputValues = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, batchedInput)
            };

            using var results = _session.Run(inputValues);
            var output = results.First();
            var outputTensor = output.AsTensor<float>().ToDenseTensor();

            if (outputTensor.Dimensions.Length == 3 && outputTensor.Dimensions[0] == 1)
            {
                var d = outputTensor.Dimensions;
                return new DenseTensor<float>(outputTensor.Buffer, new int[] { d[1], d[2] });
            }

            return outputTensor;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
