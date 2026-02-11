using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx
{
    public class OnnxDecoder : IDisposable
    {
        private readonly InferenceSession? _embSession;
        private readonly InferenceSession _decSession;
        private List<NamedOnnxValue> _pastKeyValues = new List<NamedOnnxValue>();

        public OnnxDecoder(string modelDir, SessionOptions? options = null)
        {
            var sessionOptions = options ?? new SessionOptions();

            string embPath = Path.Combine(modelDir, "embeddings.onnx");
            if (File.Exists(embPath))
            {
                _embSession = new InferenceSession(embPath, sessionOptions);
            }

            string decPath = Path.Combine(modelDir, "decoder.onnx");
            _decSession = new InferenceSession(decPath, sessionOptions);
        }

        public DenseTensor<float> EmbedToken(int tokenId)
        {
            if (_embSession == null)
            {
                 throw new InvalidOperationException("embeddings.onnx not found.");
            }

            var inputTensor = new DenseTensor<long>(new long[] { tokenId }, new int[] { 1, 1 });
            string inputName = _embSession.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using var results = _embSession.Run(inputs);
            var output = results.First().AsTensor<float>().ToDenseTensor();

            // Output [1, 1, dim]
            var d = output.Dimensions;
            int lastDim = d[d.Length - 1];

            // Return [dim]
            return new DenseTensor<float>(output.Buffer, new int[] { lastDim });
        }

        public void Prefill(DenseTensor<float> inputEmbeds, DenseTensor<float> tCond)
        {
            RunDecoder(inputEmbeds, 0, tCond);
        }

        public DenseTensor<float> ForwardOne(DenseTensor<float> embed, int pos, DenseTensor<float> tCond)
        {
             return RunDecoder(embed, pos, tCond);
        }

        private DenseTensor<float> RunDecoder(DenseTensor<float> inputEmbeds, int pos, DenseTensor<float> tCond)
        {
            var dims = inputEmbeds.Dimensions;
            DenseTensor<float> batchedInput;
            if (dims.Length == 1) // [dim]
            {
                 batchedInput = new DenseTensor<float>(inputEmbeds.Buffer, new int[] { 1, 1, dims[0] });
            }
            else if (dims.Length == 2) // [seqLen, dim]
            {
                 batchedInput = new DenseTensor<float>(inputEmbeds.Buffer, new int[] { 1, dims[0], dims[1] });
            }
            else
            {
                batchedInput = inputEmbeds;
            }

            // Reshape tCond if needed [1, dim]
            DenseTensor<float> batchedTCond = tCond;
             if (tCond.Dimensions.Length == 1)
            {
                batchedTCond = new DenseTensor<float>(tCond.Buffer, new int[] { 1, tCond.Dimensions[0] });
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds", batchedInput),
                NamedOnnxValue.CreateFromTensor("time_cond", batchedTCond)
            };

            if (_pastKeyValues != null && _pastKeyValues.Count > 0)
            {
                inputs.AddRange(_pastKeyValues);
            }

            using var results = _decSession.Run(inputs);

            // Handle KV cache updates
            var newPast = new List<NamedOnnxValue>();
            foreach(var r in results)
            {
                if (r.Name.Contains("present") || r.Name.Contains("key_values"))
                {
                    // Heuristic: if output name contains 'present', map it to 'past'
                    // If the model follows standard naming "present_key_values_0" -> "past_key_values_0"
                    string newName = r.Name.Replace("present", "past");
                    var t = r.AsTensor<float>().ToDenseTensor();
                    newPast.Add(NamedOnnxValue.CreateFromTensor(newName, t));
                }
            }
            if (newPast.Count > 0)
            {
                 _pastKeyValues = newPast;
            }

            var logitsRes = results.FirstOrDefault(r => r.Name == "logits") ?? results.First();
            var logits = logitsRes.AsTensor<float>().ToDenseTensor();

            // Extract last token logits
            var lDims = logits.Dimensions; // [1, seqLen, vocab]
            int seqLen = lDims[1];
            int vocab = lDims[2];

            var lastLogits = new float[vocab];
            int offset = (seqLen - 1) * vocab;
            logits.Buffer.Span.Slice(offset, vocab).CopyTo(lastLogits);

            return new DenseTensor<float>(lastLogits, new int[] { vocab });
        }

        public static DenseTensor<float> ComputeTimeEmbedding(float t, int dim)
        {
            float[] embData = new float[dim];
            int halfDim = dim / 2;
            float theta = 10000.0f;

            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)i / halfDim);
                float val = t * freq;
                embData[i] = MathF.Cos(val);
                embData[halfDim + i] = MathF.Sin(val);
            }
            return new DenseTensor<float>(embData, new int[] { dim });
        }

        public void Dispose()
        {
            _embSession?.Dispose();
            _decSession?.Dispose();
        }
    }
}
