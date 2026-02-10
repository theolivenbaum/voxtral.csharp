using System;
using System.IO;
using Voxtral;

namespace Voxtral.CLI
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelDir = "voxtral-model";
            string inputFile = null;

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
            }

            if (string.IsNullOrEmpty(inputFile))
            {
                Console.WriteLine("Usage: voxtral -d <model_dir> -i <input.wav>");
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
                Console.WriteLine($"Loading model from {modelDir}...");
                using var model = new VoxtralModel(modelDir);

                Console.WriteLine($"Transcribing {inputFile}...");
                string text = model.Transcribe(inputFile);

                Console.WriteLine("\nFinal Result:");
                Console.WriteLine(text);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
    }
}
