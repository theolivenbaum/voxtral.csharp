# Voxtral C# Port

This project is a pure C# port of the Voxtral Realtime 4B inference engine, originally written in C.

## Goal

The goal is to provide a dependency-free (besides standard .NET libraries and NAudio/System.Numerics.Tensors) implementation of the Voxtral model inference, capable of running on .NET 10.

## Key Features to Port

1.  **Model Loading**: Read weights from `consolidated.safetensors` (BF16/FP32).
2.  **Audio Processing**: Read WAV files, resample to 16kHz, compute Mel Spectrograms.
3.  **Tokenizer**: Tekken tokenizer implementation.
4.  **Inference Engine**:
    *   Encoder (ConvStem + Transformer)
    *   Adapter (Linear Projection)
    *   Decoder (Transformer + Time Conditioning)
5.  **CLI**: Command-line interface for transcribing audio files.

## Technical Constraints

*   Use .NET 10.
*   Use modern C# features: `Span<T>`, `SIMD`, `System.Numerics.Tensors`.
*   No external native dependencies (like Blas) - implement math in C#.
*   Use `NAudio` for WAV handling.
*   Do not focus on microphone input for now.
*   Structure: `Voxtral` (Core Library) + `Voxtral.CLI` (Console App).

## Reference

The original C code and Python reference implementation are located in the `_reference_material` folder.
