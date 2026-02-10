using System;
using System.IO;
using System.Numerics.Tensors;
using System.Text;
using System.Text.Json;
using Xunit;

namespace Voxtral.Tests
{
    public class SafetensorsReaderTests : IDisposable
    {
        private readonly string _testFile = "test.safetensors";

        public void Dispose()
        {
            if (File.Exists(_testFile))
            {
                File.Delete(_testFile);
            }
        }

        [Fact]
        public void ReadSimpleBF16Tensor()
        {
            // Create a safetensors file with one BF16 tensor
            // Tensor: "test", shape [2, 2], values [1.0, 2.0, 3.0, 4.0] (BF16)
            // BF16(1.0) = 0x3F80 >> 16 = 0x3F80
            // BF16(2.0) = 0x4000 >> 16 = 0x4000
            // BF16(3.0) = 0x4040 >> 16 = 0x4040
            // BF16(4.0) = 0x4080 >> 16 = 0x4080

            ushort[] bf16Data = new ushort[] { 0x3F80, 0x4000, 0x4040, 0x4080 };

            var header = new
            {
                test = new
                {
                    dtype = "BF16",
                    shape = new long[] { 2, 2 },
                    data_offsets = new long[] { 0, bf16Data.Length * 2 }
                }
            };

            string headerJson = JsonSerializer.Serialize(header);
            byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);
            ulong headerSize = (ulong)headerBytes.Length;

            using (var fs = new FileStream(_testFile, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(headerSize);
                bw.Write(headerBytes);
                foreach (var v in bf16Data)
                {
                    bw.Write(v);
                }
            }

            // Read it back
            using (var reader = new SafetensorsReader(_testFile))
            {
                var tensor = reader.LoadTensor("test");

                Assert.Equal(new nint[] { 2, 2 }, tensor.Lengths.ToArray());
                Assert.Equal(1.0f, tensor[0, 0]);
                Assert.Equal(2.0f, tensor[0, 1]);
                Assert.Equal(3.0f, tensor[1, 0]);
                Assert.Equal(4.0f, tensor[1, 1]);
            }
        }
    }
}
