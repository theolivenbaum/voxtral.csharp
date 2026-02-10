using System;
using System.Numerics.Tensors;
using System.Reflection;
using System.Linq;

class Program
{
    static void Main()
    {
        var type = typeof(Tensor<float>);
        Console.WriteLine("Methods on Tensor<float>:");
        foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly))
        {
             Console.WriteLine(method.ToString());
        }

        Console.WriteLine("\nMethods on Tensor (static):");
        var staticType = typeof(Tensor);
        foreach (var method in staticType.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly))
        {
             Console.WriteLine(method.ToString());
        }
    }
}
