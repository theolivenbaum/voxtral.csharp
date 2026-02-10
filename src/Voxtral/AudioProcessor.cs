using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Numerics.Tensors;

namespace Voxtral
{
    public class AudioProcessor
    {
        public const int SAMPLE_RATE = 16000;
        public const int WINDOW_SIZE = 400;
        public const int HOP_LENGTH = 160;
        public const int NUM_MEL_BINS = 128;
        public const float GLOBAL_LOG_MEL_MAX = 1.5f;

        // Streaming constants
        public const int RAW_AUDIO_LENGTH_PER_TOK = 1280;
        public const int N_LEFT_PAD_TOKENS = 32;
        public const int N_RIGHT_PAD_TOKENS = 17;

        private float[,] _melFilters;
        private float[] _hannWindow;

        public AudioProcessor()
        {
            _melFilters = ComputeMelFilters();
            _hannWindow = ComputeHannWindow(WINDOW_SIZE);
        }

        public float[] LoadAndPreprocessAudio(string filePath)
        {
            float[] audio = ReadAudio(filePath);
            return PadAudioStreaming(audio);
        }

        public Tensor<float> ComputeMelSpectrogram(float[] audio)
        {
            // STFT
            // audio is padded.
            // Number of frames:
            int nFrames = (audio.Length - WINDOW_SIZE) / HOP_LENGTH + 1;

            int numFreqBins = WINDOW_SIZE / 2 + 1; // 201

            float[,] magnitudes = new float[nFrames, numFreqBins];

            // Perform STFT
            // Using naive DFT for 400 points
            for (int i = 0; i < nFrames; i++)
            {
                int start = i * HOP_LENGTH;
                float[] frame = new float[WINDOW_SIZE];

                // Apply window
                for (int j = 0; j < WINDOW_SIZE; j++)
                {
                    if (start + j < audio.Length)
                        frame[j] = audio[start + j] * _hannWindow[j];
                    else
                        frame[j] = 0;
                }

                Complex[] dft = Dft(frame);

                for (int j = 0; j < numFreqBins; j++)
                {
                    magnitudes[i, j] = (float)dft[j].Magnitude * (float)dft[j].Magnitude; // Abs squared
                }
            }

            // Apply Mel Filterbank
            // mel_spec = mel_filters.T @ magnitudes.T  (if filters are [freq, mel])
            // In Python: mel_filters [201, 128]. magnitudes [201, frames] (after transpose of stft result)
            // Python code: mel_spec = mel_filters.T @ magnitudes  -> [128, 201] @ [201, frames] -> [128, frames]

            // Here magnitudes is [frames, 201].
            // We want [128, frames] output?
            // Tensor layout: [128, frames] usually (channels, time).

            float[] melSpecData = new float[NUM_MEL_BINS * nFrames];

            // Matrix multiplication
            // melSpec[m, t] = sum_f (melFilter[f, m] * magnitude[t, f])

            for (int t = 0; t < nFrames; t++)
            {
                for (int m = 0; m < NUM_MEL_BINS; m++)
                {
                    float sum = 0;
                    for (int f = 0; f < numFreqBins; f++)
                    {
                        sum += _melFilters[f, m] * magnitudes[t, f];
                    }

                    // Log compression
                    // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                    // log_spec = torch.maximum(log_spec, torch.tensor(GLOBAL_LOG_MEL_MAX) - 8.0)
                    // log_spec = (log_spec + 4.0) / 4.0

                    float val = Math.Max(sum, 1e-10f);
                    float logVal = (float)Math.Log10(val);
                    logVal = Math.Max(logVal, GLOBAL_LOG_MEL_MAX - 8.0f);
                    logVal = (logVal + 4.0f) / 4.0f;

                    melSpecData[m * nFrames + t] = logVal;
                }
            }

            return Tensor.Create(melSpecData, new nint[] { NUM_MEL_BINS, nFrames });
        }

        private Complex[] Dft(float[] input)
        {
            int N = input.Length;
            // We only need first N/2 + 1
            int outputSize = N / 2 + 1;
            Complex[] output = new Complex[outputSize];

            for (int k = 0; k < outputSize; k++)
            {
                double sumReal = 0;
                double sumImag = 0;
                double angleTerm = -2 * Math.PI * k / N;

                for (int n = 0; n < N; n++)
                {
                    double angle = angleTerm * n;
                    sumReal += input[n] * Math.Cos(angle);
                    sumImag += input[n] * Math.Sin(angle);
                }
                output[k] = new Complex(sumReal, sumImag);
            }
            return output;
        }

        private float[] ReadAudio(string filePath)
        {
            using var reader = new WaveFileReader(filePath);
            ISampleProvider sampler;

            if (reader.WaveFormat.Encoding == WaveFormatEncoding.Pcm)
            {
                if (reader.WaveFormat.BitsPerSample == 16)
                    sampler = new Pcm16BitToSampleProvider(reader);
                else if (reader.WaveFormat.BitsPerSample == 24)
                    sampler = new Pcm24BitToSampleProvider(reader);
                else if (reader.WaveFormat.BitsPerSample == 8)
                     sampler = new Pcm8BitToSampleProvider(reader);
                else
                    throw new NotSupportedException($"Unsupported PCM bit depth: {reader.WaveFormat.BitsPerSample}");
            }
            else if (reader.WaveFormat.Encoding == WaveFormatEncoding.IeeeFloat)
            {
                 sampler = new WaveToSampleProvider(reader);
            }
            else
            {
                throw new NotSupportedException($"Unsupported Wave encoding: {reader.WaveFormat.Encoding}");
            }

            // Resample if needed
            if (sampler.WaveFormat.SampleRate != SAMPLE_RATE)
            {
                sampler = new WdlResamplingSampleProvider(sampler, SAMPLE_RATE);
            }

            // Convert to mono if needed
            if (sampler.WaveFormat.Channels > 1)
            {
                sampler = new MultiplexingSampleProvider(new[] { sampler }, 1);
            }

            // Read all samples
            var samples = new List<float>();
            float[] buffer = new float[4096];
            int read;
            while ((read = sampler.Read(buffer, 0, buffer.Length)) > 0)
            {
                for(int i=0; i<read; i++) samples.Add(buffer[i]);
            }
            return samples.ToArray();
        }

        private float[] PadAudioStreaming(float[] audio)
        {
            int multOf = RAW_AUDIO_LENGTH_PER_TOK;
            int nSamples = audio.Length;

            // Right pad
            int alignPad = (multOf - (nSamples % multOf)) % multOf;
            int rightPad = alignPad + N_RIGHT_PAD_TOKENS * multOf;

            // Left pad
            int leftPad = N_LEFT_PAD_TOKENS * multOf;

            float[] padded = new float[leftPad + nSamples + rightPad];
            Array.Copy(audio, 0, padded, leftPad, nSamples);
            return padded;
        }

        private float[] ComputeHannWindow(int size)
        {
            float[] window = new float[size];
            for (int i = 0; i < size; i++)
            {
                window[i] = 0.5f * (1 - (float)Math.Cos(2 * Math.PI * i / size));
            }
            return window;
        }

        private float[,] ComputeMelFilters()
        {
            int numFreqBins = WINDOW_SIZE / 2 + 1; // 201
            float[] fftFreqs = LinSpace(0, SAMPLE_RATE / 2.0f, numFreqBins);
            float melMin = HertzToMel(0.0f);
            float melMax = HertzToMel(8000.0f);
            float[] melFreqs = LinSpace(melMin, melMax, NUM_MEL_BINS + 2);
            float[] filterFreqs = MelToHertz(melFreqs);

            float[,] fb = new float[numFreqBins, NUM_MEL_BINS];

            for (int m = 0; m < NUM_MEL_BINS; m++)
            {
                float f_m_minus = filterFreqs[m];
                float f_m = filterFreqs[m + 1];
                float f_m_plus = filterFreqs[m + 2];

                for (int k = 0; k < numFreqBins; k++)
                {
                    float f_k = fftFreqs[k];
                    float val = 0;

                    if (f_k >= f_m_minus && f_k <= f_m)
                    {
                        val = (f_k - f_m_minus) / (f_m - f_m_minus);
                    }
                    else if (f_k >= f_m && f_k <= f_m_plus)
                    {
                        val = (f_m_plus - f_k) / (f_m_plus - f_m);
                    }

                    fb[k, m] = val;
                }
            }

            // Enorm
            for (int m = 0; m < NUM_MEL_BINS; m++)
            {
                float enorm = 2.0f / (filterFreqs[m + 2] - filterFreqs[m]);
                for (int k = 0; k < numFreqBins; k++)
                {
                    fb[k, m] *= enorm;
                }
            }

            return fb;
        }

        private float HertzToMel(float freq)
        {
            float minLogHertz = 1000.0f;
            float minLogMel = 15.0f;
            float logStep = 27.0f / (float)Math.Log(6.4);

            if (freq >= minLogHertz)
            {
                return minLogMel + (float)Math.Log(freq / minLogHertz) * logStep;
            }
            else
            {
                return 3.0f * freq / 200.0f;
            }
        }

        private float MelToHertz(float mel)
        {
            float minLogHertz = 1000.0f;
            float minLogMel = 15.0f;
            float logStep = (float)Math.Log(6.4) / 27.0f;

            if (mel >= minLogMel)
            {
                return minLogHertz * (float)Math.Exp(logStep * (mel - minLogMel));
            }
            else
            {
                return 200.0f * mel / 3.0f;
            }
        }

        private float[] MelToHertz(float[] mels)
        {
            return mels.Select(MelToHertz).ToArray();
        }

        private float[] LinSpace(float start, float end, int count)
        {
            float[] result = new float[count];
            float step = (end - start) / (count - 1);
            for (int i = 0; i < count; i++)
            {
                result[i] = start + i * step;
            }
            return result;
        }
    }
}
