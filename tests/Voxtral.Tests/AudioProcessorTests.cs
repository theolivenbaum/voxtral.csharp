using System.IO;
using NAudio.Wave;
using Xunit;
using System.Numerics.Tensors;

namespace Voxtral.Tests
{
    public class AudioProcessorTests
    {
        [Fact]
        public void ProcessSilenceAudio()
        {
            // Create 1 second of silence
            string tempWav = "silence.wav";
            var format = new WaveFormat(16000, 1);
            using (var writer = new WaveFileWriter(tempWav, format))
            {
                byte[] data = new byte[16000 * 2]; // 16-bit
                writer.Write(data, 0, data.Length);
            }

            var processor = new AudioProcessor();
            try
            {
                var audio = processor.LoadAndPreprocessAudio(tempWav);

                // Check padding
                // 16000 samples.
                // RAW_AUDIO_LENGTH_PER_TOK = 1280.
                // 16000 % 1280 = 640.
                // alignPad = 1280 - 640 = 640.
                // Left pad = 32 * 1280 = 40960
                // Right pad = 640 + 17 * 1280 = 640 + 21760 = 22400
                // Total = 40960 + 16000 + 22400 = 79360

                Assert.Equal(79360, audio.Length);

                var mel = processor.ComputeMelSpectrogram(audio);
                // Mel shape: [128, frames]
                // Frames: (79360 - 400) / 160 + 1 = 78960 / 160 + 1 = 493 + 1 = 494

                Assert.Equal(new nint[] { 128, 494 }, mel.Lengths.ToArray());
            }
            finally
            {
                if (File.Exists(tempWav)) File.Delete(tempWav);
            }
        }
    }
}
