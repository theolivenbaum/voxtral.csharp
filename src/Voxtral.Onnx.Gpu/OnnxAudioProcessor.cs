using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;

namespace Voxtral.Onnx.Gpu
{
    public class OnnxAudioProcessor
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

        private float[] _melFiltersTransposed; // [NUM_MEL_BINS, numFreqBins] flattened
        private float[] _hannWindow;
        private float[] _dftReal; // [outputSize, WINDOW_SIZE] flattened
        private float[] _dftImag; // [outputSize, WINDOW_SIZE] flattened

        public OnnxAudioProcessor()
        {
            _hannWindow = ComputeHannWindow(WINDOW_SIZE);
            InitializeDftMatrices();
            InitializeMelFilters();
        }

        private void InitializeDftMatrices()
        {
            int N = WINDOW_SIZE;
            int outputSize = N / 2 + 1;

            _dftReal = new float[outputSize * N];
            _dftImag = new float[outputSize * N];

            // Precompute DFT matrices
            // We store them as [outputSize, N] so they can be used as weights in Linear (which expects [N_out, K_in])
            // DFT formula: X[k] = sum(x[n] * exp(-j * 2 * pi * k * n / N))
            // Real part weight: cos(-2*pi*k*n/N) = cos(2*pi*k*n/N)
            // Imag part weight: sin(-2*pi*k*n/N) = -sin(2*pi*k*n/N)

            for (int k = 0; k < outputSize; k++)
            {
                double angleTerm = -2 * Math.PI * k / N;
                for (int n = 0; n < N; n++)
                {
                    double angle = angleTerm * n;
                    // k is row, n is col
                    int idx = k * N + n;
                    _dftReal[idx] = (float)Math.Cos(angle);
                    _dftImag[idx] = (float)Math.Sin(angle);
                }
            }
        }

        private void InitializeMelFilters()
        {
            float[,] fb = ComputeMelFilters(); // [numFreqBins, NUM_MEL_BINS]
            int numFreqBins = fb.GetLength(0);
            int numMelBins = fb.GetLength(1);

            // Transpose to [NUM_MEL_BINS, numFreqBins] flattened
            _melFiltersTransposed = new float[numMelBins * numFreqBins];

            for (int m = 0; m < numMelBins; m++)
            {
                for (int f = 0; f < numFreqBins; f++)
                {
                    _melFiltersTransposed[m * numFreqBins + f] = fb[f, m];
                }
            }
        }

        public float[] LoadAndPreprocessAudio(string filePath)
        {
            float[] audio = ReadAudio(filePath);
            return PadAudioStreaming(audio);
        }

        public DenseTensor<float> ComputeMelSpectrogram(float[] audio)
        {
            // Center padding for STFT
            int pad = WINDOW_SIZE / 2;
            float[] paddedAudio = new float[audio.Length + 2 * pad];

            // Reflect left
            for (int i = 0; i < pad; i++)
            {
                int srcIdx = pad - 1 - i;
                paddedAudio[i] = (srcIdx < audio.Length) ? audio[srcIdx] : 0;
            }

            // Copy center
            Array.Copy(audio, 0, paddedAudio, pad, audio.Length);

            // Reflect right
            for (int i = 0; i < pad; i++)
            {
                int srcIdx = audio.Length - 2 - i;
                paddedAudio[pad + audio.Length + i] = (srcIdx >= 0) ? audio[srcIdx] : 0;
            }

            // Number of frames
            int nFrames = (paddedAudio.Length - WINDOW_SIZE) / HOP_LENGTH + 1;
            nFrames -= 1; // Drop last frame

            if (nFrames <= 0) return new DenseTensor<float>(new float[0], new int[] { NUM_MEL_BINS, 0 });

            int numFreqBins = WINDOW_SIZE / 2 + 1;

            // Prepare batch of frames
            float[] allFrames = new float[nFrames * WINDOW_SIZE];

            // Fill frames
            // We can parallelize this loop if needed, but it's memory bound mostly
            Parallel.For(0, nFrames, i =>
            {
                int start = i * HOP_LENGTH;
                int offset = i * WINDOW_SIZE;
                for (int j = 0; j < WINDOW_SIZE; j++)
                {
                    if (start + j < paddedAudio.Length)
                        allFrames[offset + j] = paddedAudio[start + j] * _hannWindow[j];
                    else
                        allFrames[offset + j] = 0;
                }
            });

            // Compute DFT using Matrix Multiplication
            float[] dftReal = new float[nFrames * numFreqBins];
            float[] dftImag = new float[nFrames * numFreqBins];

            // Real part
            OnnxTensorOperations.Linear(allFrames, _dftReal, ReadOnlySpan<float>.Empty, dftReal, nFrames, numFreqBins, WINDOW_SIZE);
            // Imag part
            OnnxTensorOperations.Linear(allFrames, _dftImag, ReadOnlySpan<float>.Empty, dftImag, nFrames, numFreqBins, WINDOW_SIZE);

            // Compute Magnitudes Squared
            float[] magnitudes = new float[nFrames * numFreqBins];
            // magnitudes = real^2 + imag^2
            // Vectorize this
            int len = magnitudes.Length;
            int vecSize = Vector<float>.Count;
            int limit = len - (len % vecSize);

            for (int i = 0; i < limit; i += vecSize)
            {
                Vector<float> vr = new Vector<float>(dftReal.AsSpan(i));
                Vector<float> vi = new Vector<float>(dftImag.AsSpan(i));
                Vector<float> magSq = vr * vr + vi * vi;
                magSq.CopyTo(magnitudes.AsSpan(i));
            }
            for (int i = limit; i < len; i++)
            {
                magnitudes[i] = dftReal[i] * dftReal[i] + dftImag[i] * dftImag[i];
            }

            // Apply Mel Filterbank using Matrix Multiplication
            // magnitudes: [nFrames, numFreqBins]
            // melFilters: [NUM_MEL_BINS, numFreqBins] (transposed)
            // Output: [nFrames, NUM_MEL_BINS]

            float[] melSpecDataTemp = new float[nFrames * NUM_MEL_BINS];
            OnnxTensorOperations.Linear(magnitudes, _melFiltersTransposed, ReadOnlySpan<float>.Empty, melSpecDataTemp, nFrames, NUM_MEL_BINS, numFreqBins);

            // Transpose output to [NUM_MEL_BINS, nFrames] AND apply log compression
            // The result structure required is [NUM_MEL_BINS, nFrames] (column-major effectively if viewed as [nFrames, MEL])
            // But Tensor needs row-major of [NUM_MEL_BINS, nFrames].
            // So we need to transpose melSpecDataTemp.
            // melSpecDataTemp is [nFrames, NUM_MEL_BINS] (row-major).
            // We need [NUM_MEL_BINS, nFrames].

            float[] finalMelSpec = new float[NUM_MEL_BINS * nFrames];

            // Loop over nFrames (rows of temp) and NUM_MEL_BINS (cols of temp)
            // Destination: dest[col * nFrames + row] = process(src[row * NUM_MEL_BINS + col])

            Parallel.For(0, nFrames, t =>
            {
                for (int m = 0; m < NUM_MEL_BINS; m++)
                {
                    float val = melSpecDataTemp[t * NUM_MEL_BINS + m];

                    // Log compression
                    val = Math.Max(val, 1e-10f);
                    float logVal = (float)Math.Log10(val);
                    logVal = Math.Max(logVal, GLOBAL_LOG_MEL_MAX - 8.0f);
                    logVal = (logVal + 4.0f) / 4.0f;

                    // Transpose write
                    finalMelSpec[m * nFrames + t] = logVal;
                }
            });

            return new DenseTensor<float>(finalMelSpec, new int[] { NUM_MEL_BINS, nFrames });
        }

        // Keep helper methods ...
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

            if (sampler.WaveFormat.SampleRate != SAMPLE_RATE)
            {
                sampler = new WdlResamplingSampleProvider(sampler, SAMPLE_RATE);
            }

            if (sampler.WaveFormat.Channels > 1)
            {
                sampler = new MultiplexingSampleProvider(new[] { sampler }, 1);
            }

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

            int alignPad = (multOf - (nSamples % multOf)) % multOf;
            int rightPad = alignPad + N_RIGHT_PAD_TOKENS * multOf;
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
