using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Voxtral;

namespace Voxtral.Onnx.Gpu
{
    public class VoxtralOnnxGpuModel : IVoxtralModel
    {
        private InferenceSession? _encoderSession;
        private InferenceSession? _adapterSession;
        private InferenceSession? _decoderSession;
        private Tokenizer _tokenizer;

        public VoxtralOnnxGpuModel(string modelDir)
        {
            var sw = Stopwatch.StartNew();
            Console.WriteLine("Loading model (ONNX GPU Backend)...");

            _tokenizer = new Tokenizer(modelDir);

            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

            // GPU Execution Provider Configuration
            try
            {
                bool isX64 = RuntimeInformation.ProcessArchitecture == Architecture.X64;

                if ((RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) && isX64)
                {
                    // Attempt to use CUDA
                    options.AppendExecutionProvider_CUDA();
                    Console.WriteLine("CUDA Execution Provider initialized.");
                }
                else
                {
                    Console.WriteLine($"OS/Arch not supported for CUDA (Linux/Windows X64 only). Fallback to CPU. Arch: {RuntimeInformation.ProcessArchitecture}");
                }
            }
            catch (Exception ex)
            {
                 Console.WriteLine($"WARNING: Failed to initialize CUDA execution provider: {ex.Message}. Fallback to CPU.");
            }

            // Load sessions
            string encPath = Path.Combine(modelDir, "encoder.onnx");
            string adpPath = Path.Combine(modelDir, "adapter.onnx");
            string decPath = Path.Combine(modelDir, "decoder.onnx");

            if (!File.Exists(encPath)) throw new FileNotFoundException($"Model file not found: {encPath}", encPath);
            if (!File.Exists(adpPath)) throw new FileNotFoundException($"Model file not found: {adpPath}", adpPath);
            if (!File.Exists(decPath)) throw new FileNotFoundException($"Model file not found: {decPath}", decPath);

            _encoderSession = new InferenceSession(encPath, options);
            _adapterSession = new InferenceSession(adpPath, options);
            _decoderSession = new InferenceSession(decPath, options);

            Console.WriteLine($"Loaded models in {sw.Elapsed.TotalSeconds:n0}s...");
        }

        public void Dispose()
        {
            _encoderSession?.Dispose();
            _adapterSession?.Dispose();
            _decoderSession?.Dispose();
        }

        public string Transcribe(string wavPath)
        {
            if (_encoderSession == null || _adapterSession == null || _decoderSession == null)
                throw new InvalidOperationException("Sessions not initialized.");

            var sw = Stopwatch.StartNew();
            Console.WriteLine("Processing audio...");

            // 1. Audio Processing
            var processor = new AudioProcessor();
            var audio = processor.LoadAndPreprocessAudio(wavPath);

            // Returns System.Numerics.Tensors.Tensor<float>
            var melTensor = processor.ComputeMelSpectrogram(audio);

            // Convert to ONNX Tensor [1, NUM_MEL_BINS, nFrames]
            int numMelBins = (int)melTensor.Lengths[0];
            int nFrames = (int)melTensor.Lengths[1];

            // Flatten data from System.Numerics.Tensors.Tensor<float>
            // Note: Since we don't have direct Span access to underlying storage guaranteed, we iterate.
            float[] melData = new float[melTensor.FlattenedLength];
            int i = 0;
            foreach(var val in melTensor) { melData[i++] = val; }

            // Create ONNX Tensor
            var inputOrt = new DenseTensor<float>(melData, new int[] { 1, numMelBins, nFrames });

            // 2. Encoder
            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("mel", inputOrt)
            };

            using var encoderResults = _encoderSession.Run(encoderInputs);
            var audioEmbeds = encoderResults.First().AsTensor<float>(); // [1, seqLen, hidden]

            // 3. Adapter
            // Need to convert audioEmbeds (Tensor<float>) back to NamedOnnxValue input
            // AsTensor returns a view. To use it as input, we can just pass it if the graph allows.
            // Or create a DenseTensor from it.
            // For safety, let's assume we can pass it directly or copy it.
            // audioEmbeds is of type Tensor<float> (OnnxRuntime).

            var adapterInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("audio_embeds", audioEmbeds.ToDenseTensor())
            };

            using var adapterResults = _adapterSession.Run(adapterInputs);
            var adaptedEmbeds = adapterResults.First().AsTensor<float>();

            // 4. Decoder
            List<long> tokens = new List<long> { 1 }; // BOS

            // Clone adapted embeds to keep them alive if needed, or just use ToDenseTensor
            var encHiddenTensor = adaptedEmbeds.ToDenseTensor();

            Console.WriteLine("Generating...");

            for(int step = 0; step < 200; step++) // Max steps limit
            {
                var inputIdsTensor = new DenseTensor<long>(tokens.ToArray(), new int[] { 1, tokens.Count });

                var decoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                    NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encHiddenTensor)
                };

                using var decoderResults = _decoderSession.Run(decoderInputs);
                var logits = decoderResults.First().AsTensor<float>(); // [1, seq, vocab]

                // Argmax on last token
                int vocabSize = (int)logits.Dimensions[2];
                int seqLen = (int)logits.Dimensions[1];

                // We want the last token's logits: [0, seqLen-1, :]
                // Tensor<T> allows indexer access.

                float maxVal = float.NegativeInfinity;
                int maxIdx = 0;

                for (int v = 0; v < vocabSize; v++)
                {
                    float val = logits[0, seqLen - 1, v];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxIdx = v;
                    }
                }

                if (maxIdx == 2) break; // EOS (assuming 2 is EOS)
                tokens.Add(maxIdx);

                // Print partial?
                // Console.Write(maxIdx + " ");
            }

            Console.WriteLine();
            return _tokenizer.Decode(tokens.Select(t => (int)t).ToArray());
        }
    }
}
