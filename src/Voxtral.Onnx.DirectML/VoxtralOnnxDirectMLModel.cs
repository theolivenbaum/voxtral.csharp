using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using Voxtral;

namespace Voxtral.Onnx.DirectML
{
    public class VoxtralOnnxDirectMLModel : IVoxtralModel
    {
        private OnnxSafetensorsReader _reader;
        private OnnxEncoder _encoder;
        private OnnxAdapter _adapter;
        private OnnxDecoder _decoder;
        private Tokenizer _tokenizer;

        public VoxtralOnnxDirectMLModel(string modelDir)
        {
            var sw = Stopwatch.StartNew();
            Console.WriteLine("Loading model (ONNX DirectML Backend)...");
            string safetensorsPath = Path.Combine(modelDir, "consolidated.safetensors");
            _reader = new OnnxSafetensorsReader(safetensorsPath);

            _encoder = new OnnxEncoder(_reader);
            Console.WriteLine($"Loaded encoder in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            _adapter = new OnnxAdapter(_reader);
            Console.WriteLine($"Loaded reader in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            _decoder = new OnnxDecoder(_reader);
            Console.WriteLine($"Loaded decoder in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            _tokenizer = new Tokenizer(modelDir);
            Console.WriteLine($"Loaded tokenizer in {sw.Elapsed.TotalSeconds:n0}s...");
        }

        public void Dispose()
        {
            _reader?.Dispose();
        }

        public string Transcribe(string wavPath)
        {
            // Audio
            var sw = Stopwatch.StartNew();
            Console.WriteLine("Processing audio...");
            var processor = new OnnxAudioProcessor();
            var audio = processor.LoadAndPreprocessAudio(wavPath);
            var mel = processor.ComputeMelSpectrogram(audio);
            Console.WriteLine($"Loaded audio in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();

            // Encoder
            Console.WriteLine("Running Encoder...");
            var encOut = _encoder.Forward(mel, out int encSeqLen);
            Console.WriteLine($"Encoded audio in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            // Adapter
            Console.WriteLine("Running Adapter...");
            var adapterOut = _adapter.Forward(encOut);
            Console.WriteLine($"Adapted audio in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            int nAudio = encSeqLen / 4; // Downsample factor

            // Decoder
            Console.WriteLine("Running Decoder...");
            int nLeftPad = OnnxAudioProcessor.N_LEFT_PAD_TOKENS;
            int nDelay = 6;

            List<int> promptIds = new List<int> { 1 }; // BOS
            for (int i = 0; i < nLeftPad + nDelay; i++) promptIds.Add(32); // STREAMING_PAD

            int L = promptIds.Count;

            if (L > nAudio) throw new Exception("Audio too short");

            // Combine embeddings
            float[] prefixEmbedsData = new float[L * 3072];
            DenseTensor<float> prefixEmbeds = new DenseTensor<float>(prefixEmbedsData, new int[] { L, 3072 });

            var prefixSpan = prefixEmbeds.Buffer.Span;
            var adapterSpan = adapterOut.Buffer.Span;

            for (int i = 0; i < L; i++)
            {
                var txtEmb = _decoder.EmbedToken(promptIds[i]);
                var txtEmbSpan = txtEmb.Buffer.Span;
                var audEmb = adapterSpan.Slice(i * 3072, 3072);

                for (int j = 0; j < 3072; j++)
                {
                    prefixSpan[i * 3072 + j] = txtEmbSpan[j] + audEmb[j];
                }
            }

            // Time Cond
            DenseTensor<float> tCond = OnnxDecoder.ComputeTimeEmbedding(nDelay, 3072);

            // Prefill
            if (L > 1)
            {
                float[] prefillInputData = new float[(L - 1) * 3072];
                // Copy from prefixEmbeds
                prefixSpan.Slice(0, (L - 1) * 3072).CopyTo(prefillInputData);

                DenseTensor<float> prefillInput = new DenseTensor<float>(prefillInputData, new int[] { L - 1, 3072 });
                _decoder.Prefill(prefillInput, tCond);
            }

            // Generate first token
            float[] lastEmbedData = new float[3072];
            prefixSpan.Slice((L - 1) * 3072, 3072).CopyTo(lastEmbedData);
            DenseTensor<float> lastEmbed = new DenseTensor<float>(lastEmbedData, new int[] { 3072 });

            var logits = _decoder.ForwardOne(lastEmbed, L - 1, tCond);
            int token = ArgMax(logits);

            var generated = new List<int> { token };
            var sb = new StringBuilder();
            Console.Write(_tokenizer.Decode(token));

            // Generate rest
            for (int pos = L; pos < nAudio; pos++)
            {
                if (token == 2) break; // EOS

                var txtEmb = _decoder.EmbedToken(token);
                var txtEmbSpan = txtEmb.Buffer.Span;

                // adapterSpan is valid here (captured from adapterOut)
                var audEmb = adapterSpan.Slice(pos * 3072, 3072);

                float[] embedData = new float[3072];
                // Manual add
                for (int j = 0; j < 3072; j++) embedData[j] = txtEmbSpan[j] + audEmb[j];

                DenseTensor<float> embed = new DenseTensor<float>(embedData, new int[] { 3072 });

                logits = _decoder.ForwardOne(embed, pos, tCond);
                token = ArgMax(logits);
                generated.Add(token);

                Console.Write(_tokenizer.Decode(token));
            }
            Console.WriteLine();

            var result = _tokenizer.Decode(generated.ToArray());

            Console.WriteLine($"Decoded audio in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();

            return result;
        }

        private int ArgMax(DenseTensor<float> logits)
        {
            var span = logits.Buffer.Span;
            float maxVal = float.NegativeInfinity;
            int maxIdx = -1;
            for (int i = 0; i < span.Length; i++)
            {
                if (span[i] > maxVal)
                {
                    maxVal = span[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
    }
}
