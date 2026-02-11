using System;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Voxtral;
using Voxtral.Onnx.DirectML;

namespace Voxtral.DirectML.CLI
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelDir = "voxtral-model";
            string inputFile = null;
            string backend = "dotnet";

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-d" || args[i] == "--model" || args[i] == "-m")
                {
                    if (i + 1 < args.Length) modelDir = args[++i];
                }
                else if (args[i] == "-i" || args[i] == "--input")
                {
                    if (i + 1 < args.Length) inputFile = args[++i];
                }
                else if (args[i] == "--backend" || args[i] == "-b")
                {
                    if (i + 1 < args.Length) backend = args[++i].ToLower();
                }
            }

            if (string.IsNullOrEmpty(inputFile))
            {
                Console.WriteLine("Usage: voxtral-dml -d <model_dir> -i <input.wav> [--backend <dotnet|onnx|directml>]");
                return;
            }

            if (!Directory.Exists(modelDir))
            {
                Console.WriteLine($"Error: Model directory '{modelDir}' not found.");
                return;
            }

            if (!File.Exists(inputFile))
            {
                Console.WriteLine($"Error: Input file '{inputFile}' not found.");
                return;
            }

            try
            {
                Console.WriteLine($"Loading model from {modelDir} (Backend: {backend})...");

                IVoxtralModel model;
                if (backend == "onnx" || backend == "directml")
                {
                    model = new VoxtralOnnxDirectMLModel(modelDir);
                }
                else
                {
                    model = new VoxtralModel(modelDir);
                }

                using (model)
                {
                    Console.WriteLine($"Transcribing {inputFile}...");
                    string text = model.Transcribe(inputFile);

                    Console.WriteLine("\nFinal Result:");
                    Console.WriteLine(text);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
    }
}
