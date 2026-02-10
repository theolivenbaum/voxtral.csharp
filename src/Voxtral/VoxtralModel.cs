using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Numerics.Tensors;
using System.Text;

namespace Voxtral
{
    public class VoxtralModel : IDisposable
    {
        private SafetensorsReader _reader;
        private Encoder _encoder;
        private Adapter _adapter;
        private Decoder _decoder;
        private Tokenizer _tokenizer;

        public VoxtralModel(string modelDir)
        {
            var sw = Stopwatch.StartNew();
            Console.WriteLine("Loading model...");
            string safetensorsPath = Path.Combine(modelDir, "consolidated.safetensors");
            _reader = new SafetensorsReader(safetensorsPath);

            _encoder = new Encoder(_reader);
            Console.WriteLine($"Loaded encoder in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            _adapter = new Adapter(_reader);
            Console.WriteLine($"Loaded reader in {sw.Elapsed.TotalSeconds:n0}s..."); sw.Restart();
            _decoder = new Decoder(_reader);
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
            var processor = new AudioProcessor();
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
            int nLeftPad = AudioProcessor.N_LEFT_PAD_TOKENS;
            int nDelay = 6;

            List<int> promptIds = new List<int> { 1 }; // BOS
            for (int i = 0; i < nLeftPad + nDelay; i++) promptIds.Add(32); // STREAMING_PAD

            int L = promptIds.Count;

            if (L > nAudio) throw new Exception("Audio too short");

            // Combine embeddings
            float[] prefixEmbedsData = new float[L * 3072];
            Tensor<float> prefixEmbeds = Tensor.Create(prefixEmbedsData, new nint[] { L, 3072 });

            var prefixSpan = prefixEmbeds.AsSpan();
            var adapterSpan = adapterOut.AsSpan();

            for (int i = 0; i < L; i++)
            {
                var txtEmb = _decoder.EmbedToken(promptIds[i]);
                var txtEmbSpan = txtEmb.AsSpan();
                var audEmb = adapterSpan.Slice(i * 3072, 3072);

                for (int j = 0; j < 3072; j++)
                {
                    prefixSpan[i * 3072 + j] = txtEmbSpan[j] + audEmb[j];
                }
            }

            // Time Cond
            Tensor<float> tCond = Decoder.ComputeTimeEmbedding(nDelay, 3072);

            // Prefill
            if (L > 1)
            {
                float[] prefillInputData = new float[(L - 1) * 3072];
                // Copy from prefixEmbeds
                prefixSpan.Slice(0, (L - 1) * 3072).CopyTo(prefillInputData);

                Tensor<float> prefillInput = Tensor.Create(prefillInputData, new nint[] { L - 1, 3072 });
                _decoder.Prefill(prefillInput, tCond);
            }

            // Generate first token
            float[] lastEmbedData = new float[3072];
            prefixSpan.Slice((L - 1) * 3072, 3072).CopyTo(lastEmbedData);
            Tensor<float> lastEmbed = Tensor.Create(lastEmbedData, new nint[] { 3072 });

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
                var txtEmbSpan = txtEmb.AsSpan();

                // adapterSpan is valid here (captured from adapterOut)
                var audEmb = adapterSpan.Slice(pos * 3072, 3072);

                float[] embedData = new float[3072];
                // Manual add
                for (int j = 0; j < 3072; j++) embedData[j] = txtEmbSpan[j] + audEmb[j];

                Tensor<float> embed = Tensor.Create(embedData, new nint[] { 3072 });

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

        private int ArgMax(Tensor<float> logits)
        {
            var span = logits.AsSpan();
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
