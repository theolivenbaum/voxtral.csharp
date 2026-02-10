using System;
using System.Numerics.Tensors;
using System.Linq;

class Program {
    static void Main() {
        float[] data = new float[10];
        for(int i=0;i<10;i++) data[i] = i;
        nint[] shape = new nint[] { 2, 5 };

        // Tensor.Create is what SafetensorsReader uses.
        Tensor<float> t = Tensor.Create(data, shape);

        Console.WriteLine($"Rank: {t.Rank}");
        Console.WriteLine($"Shape: {string.Join(", ", t.Shape.Select(x => x.ToString()))}"); // Flatten? or how to access shape?
        // t.Shape is ReadOnlySpan<nint>

        // Access element
        // t[0, 1] ?

        // AsSpan ?
        // Does it have AsSpan?
        // It might be t.AsReadOnlySpan() or similar?

        // Let's try to compile this and see errors if any.
        // I'll try a few things.

        // t.Buffer?
    }
}
