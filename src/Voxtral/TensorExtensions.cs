using System;
using System.Numerics.Tensors;

namespace Voxtral
{
    public static class TensorExtensions
    {
        public static Span<T> AsSpan<T>(this Tensor<T> tensor)
        {
            TensorSpan<T> ts = tensor;

            int rank = tensor.Rank;
            Span<nint> start = rank <= 4 ? stackalloc nint[rank] : new nint[rank];
            start.Clear();

            return ts.GetSpan(start, (int)ts.FlattenedLength);
        }
    }
}
