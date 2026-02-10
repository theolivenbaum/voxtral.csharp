using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Voxtral.Onnx
{
    public class OnnxSafetensorsReader : IDisposable
    {
        private readonly FileStream _fileStream;
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly long _dataStart;
        private readonly Dictionary<string, TensorInfo> _tensorInfos;

        public OnnxSafetensorsReader(string filePath)
        {
            _fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);

            // Read header size (8 bytes, uint64, little endian)
            byte[] sizeBuffer = new byte[8];
            _fileStream.ReadExactly(sizeBuffer);
            long headerSize = (long)BitConverter.ToUInt64(sizeBuffer);

            // Read header
            byte[] headerBuffer = new byte[headerSize];
            _fileStream.ReadExactly(headerBuffer);
            string headerJson = Encoding.UTF8.GetString(headerBuffer);

            // Parse header
            _tensorInfos = JsonSerializer.Deserialize<Dictionary<string, TensorInfo>>(headerJson);

            _dataStart = 8 + headerSize;

            // Create memory mapped file for efficient access
            _mmf = MemoryMappedFile.CreateFromFile(_fileStream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, false);
            _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        public DenseTensor<float> LoadTensor(string name)
        {
            if (!_tensorInfos.TryGetValue(name, out var info))
            {
                throw new KeyNotFoundException($"Tensor '{name}' not found in safetensors file.");
            }

            long offset = _dataStart + info.data_offsets[0];
            long length = info.data_offsets[1] - info.data_offsets[0];

            // Calculate number of elements
            long numElements = 1;
            foreach (var dim in info.shape)
            {
                numElements *= dim;
            }

            // Read data
            float[] data = new float[numElements];

            if (info.dtype == "BF16")
            {
                ReadBFloat16(offset, data);
            }
            else if (info.dtype == "F32")
            {
                ReadFloat32(offset, data);
            }
            else
            {
                 throw new NotSupportedException($"Dtype '{info.dtype}' is not supported yet.");
            }

            // Create DenseTensor
            // Microsoft.ML.OnnxRuntime.Tensors.DenseTensor expects ReadOnlySpan<int> for dimensions.
            int[] dimensions = info.shape.Select(x => (int)x).ToArray();

            return new DenseTensor<float>(data, dimensions);
        }

        private unsafe void ReadBFloat16(long offset, float[] destination)
        {
            byte* basePtr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            try
            {
                ushort* src = (ushort*)(basePtr + offset);
                fixed (float* dst = destination)
                {
                    uint* dstAsUint = (uint*)dst;
                    for (int i = 0; i < destination.Length; i++)
                    {
                        // Convert BF16 to FP32: shift left 16 bits
                        dstAsUint[i] = (uint)src[i] << 16;
                    }
                }
            }
            finally
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }
        }

        private unsafe void ReadFloat32(long offset, float[] destination)
        {
             byte* basePtr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            try
            {
                float* src = (float*)(basePtr + offset);
                // Can use Span.Copy if we cast safely, or just loop
                // Direct copy is faster
                Marshal.Copy((IntPtr)src, destination, 0, destination.Length);
            }
            finally
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }
        }

        public void Dispose()
        {
            _accessor?.Dispose();
            _mmf?.Dispose();
            _fileStream?.Dispose();
        }

        private class TensorInfo
        {
            public string dtype { get; set; }
            public long[] shape { get; set; }
            public long[] data_offsets { get; set; }
        }
    }
}
