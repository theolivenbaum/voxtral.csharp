using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Numerics.Tensors;
using System.Threading.Tasks;

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

        private Tensor<float> _melFiltersTransposed; // [NUM_MEL_BINS, numFreqBins]
        private float[] _hannWindow;
        private Tensor<float> _dftReal; // [outputSize, WINDOW_SIZE]
        private Tensor<float> _dftImag; // [outputSize, WINDOW_SIZE]

        public AudioProcessor()
        {
            _hannWindow = ComputeHannWindow(WINDOW_SIZE);
            InitializeDftMatrices();
            InitializeMelFilters();
        }

        private void InitializeDftMatrices()
        {
            int N = WINDOW_SIZE;
            int outputSize = N / 2 + 1;

            float[] dftRealData = new float[outputSize * N];
            float[] dftImagData = new float[outputSize * N];

            for (int k = 0; k < outputSize; k++)
            {
                double angleTerm = -2 * Math.PI * k / N;
                for (int n = 0; n < N; n++)
                {
                    double angle = angleTerm * n;
                    int idx = k * N + n;
                    dftRealData[idx] = (float)Math.Cos(angle);
                    dftImagData[idx] = (float)Math.Sin(angle);
                }
            }

            _dftReal = Tensor.Create(dftRealData, new nint[] { outputSize, N });
            _dftImag = Tensor.Create(dftImagData, new nint[] { outputSize, N });
        }

        private void InitializeMelFilters()
        {
            float[,] fb = ComputeMelFilters();
            int numFreqBins = fb.GetLength(0);
            int numMelBins = fb.GetLength(1);

            float[] melFiltersData = new float[numMelBins * numFreqBins];

            for (int m = 0; m < numMelBins; m++)
            {
                for (int f = 0; f < numFreqBins; f++)
                {
                    melFiltersData[m * numFreqBins + f] = fb[f, m];
                }
            }

            _melFiltersTransposed = Tensor.Create(melFiltersData, new nint[] { numMelBins, numFreqBins });
        }

        public float[] LoadAndPreprocessAudio(string filePath)
        {
            float[] audio = ReadAudio(filePath);
            return PadAudioStreaming(audio);
        }

        public Tensor<float> ComputeMelSpectrogram(float[] audio)
        {
            int pad = WINDOW_SIZE / 2;
            float[] paddedAudio = new float[audio.Length + 2 * pad];

            for (int i = 0; i < pad; i++)
            {
                int srcIdx = pad - 1 - i;
                paddedAudio[i] = (srcIdx < audio.Length) ? audio[srcIdx] : 0;
            }

            Array.Copy(audio, 0, paddedAudio, pad, audio.Length);

            for (int i = 0; i < pad; i++)
            {
                int srcIdx = audio.Length - 2 - i;
                paddedAudio[pad + audio.Length + i] = (srcIdx >= 0) ? audio[srcIdx] : 0;
            }

            int nFrames = (paddedAudio.Length - WINDOW_SIZE) / HOP_LENGTH + 1;
            nFrames -= 1;

            if (nFrames <= 0) return Tensor.Create(new float[0], new nint[] { NUM_MEL_BINS, 0 });

            int numFreqBins = WINDOW_SIZE / 2 + 1;

            float[] allFrames = new float[nFrames * WINDOW_SIZE];

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

            Tensor<float> allFramesTensor = Tensor.Create(allFrames, new nint[] { nFrames, WINDOW_SIZE });

            float[] dftRealData = new float[nFrames * numFreqBins];
            float[] dftImagData = new float[nFrames * numFreqBins];

            Tensor<float> dftRealTensor = Tensor.Create(dftRealData, new nint[] { nFrames, numFreqBins });
            Tensor<float> dftImagTensor = Tensor.Create(dftImagData, new nint[] { nFrames, numFreqBins });

            TensorOperations.Linear(allFramesTensor, _dftReal, null, dftRealTensor);
            TensorOperations.Linear(allFramesTensor, _dftImag, null, dftImagTensor);

            TensorSpan<float> realSpan = dftRealTensor;
            TensorSpan<float> imagSpan = dftImagTensor;

            // Compute magnitudes
            // magnitudes = real^2 + imag^2
            // We can reuse buffers or create new one.
            float[] magnitudesData = new float[nFrames * numFreqBins];
            Tensor<float> magnitudesTensor = Tensor.Create(magnitudesData, new nint[] { nFrames, numFreqBins });
            TensorSpan<float> magSpan = magnitudesTensor;

            Tensor.Multiply<float>(realSpan, realSpan, realSpan);
            Tensor.Multiply<float>(imagSpan, imagSpan, imagSpan);
            Tensor.Add<float>(realSpan, imagSpan, magSpan);

            float[] melSpecDataTemp = new float[nFrames * NUM_MEL_BINS];
            Tensor<float> melSpecTempTensor = Tensor.Create(melSpecDataTemp, new nint[] { nFrames, NUM_MEL_BINS });

            TensorOperations.Linear(magnitudesTensor, _melFiltersTransposed, null, melSpecTempTensor);

            float[] finalMelSpec = new float[NUM_MEL_BINS * nFrames];

            Parallel.For(0, nFrames, t =>
            {
                var melSpecTempSpan = melSpecTempTensor.AsSpan();
                for (int m = 0; m < NUM_MEL_BINS; m++)
                {
                    float val = melSpecTempSpan[t * NUM_MEL_BINS + m];

                    val = Math.Max(val, 1e-10f);
                    float logVal = (float)Math.Log10(val);
                    logVal = Math.Max(logVal, GLOBAL_LOG_MEL_MAX - 8.0f);
                    logVal = (logVal + 4.0f) / 4.0f;

                    finalMelSpec[m * nFrames + t] = logVal;
                }
            });

            return Tensor.Create(finalMelSpec, new nint[] { NUM_MEL_BINS, nFrames });
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
