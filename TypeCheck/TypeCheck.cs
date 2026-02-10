using System;
using System.Numerics.Tensors;

class Program {
    static void Main() {
        float[] data = new float[10];
        var t = Tensor.Create(data, new nint[] { 2, 5 });
        var s = t.AsReadOnlyTensorSpan();
        // Check IsContiguous property?
        // It might be IsContiguous() method or property.
    }
}
