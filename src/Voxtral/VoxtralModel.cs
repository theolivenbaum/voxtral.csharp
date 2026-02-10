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
            float[] prefixEmbeds = new float[L * 3072];
            var adapterSpan = adapterOut.AsSpan();

            for (int i = 0; i < L; i++)
            {
                var txtEmb = _decoder.EmbedToken(promptIds[i]);
                var audEmb = adapterSpan.Slice(i * 3072, 3072);

                for (int j = 0; j < 3072; j++)
                {
                    prefixEmbeds[i * 3072 + j] = txtEmb[j] + audEmb[j];
                }
            }

            // Time Cond
            float[] tCond = Decoder.ComputeTimeEmbedding(nDelay, 3072);

            // Prefill
            if (L > 1)
            {
                float[] prefillInput = new float[(L - 1) * 3072];
                Array.Copy(prefixEmbeds, 0, prefillInput, 0, (L - 1) * 3072);
                _decoder.Prefill(prefillInput, tCond);
            }

            // Generate first token
            float[] lastEmbed = new float[3072];
            Array.Copy(prefixEmbeds, (L - 1) * 3072, lastEmbed, 0, 3072);

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
                var audEmb = adapterSpan.Slice(pos * 3072, 3072);

                float[] embed = new float[3072];
                for (int j = 0; j < 3072; j++) embed[j] = txtEmb[j] + audEmb[j];

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

        private int ArgMax(float[] logits)
        {
            float maxVal = float.NegativeInfinity;
            int maxIdx = -1;
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxVal)
                {
                    maxVal = logits[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
    }
}
